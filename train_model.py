"""
train_model.py

Usage:
  python train_model.py --train traindata.csv --test testdata.csv [--no-embeddings]

This script trains classifiers on TF-IDF and (optionally) SentenceTransformer embeddings,
creates voting ensembles per-branch and a stacked meta-classifier, evaluates on the
provided test set, and saves artifacts under ./models/.
"""
import argparse
import os
from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report


def combine_text(df):
    return (df['topic'].fillna('') + ' ||| ' + df['answer'].fillna('')).astype(str)


def evaluate(y_true, probs, thr=0.5):
    preds = (probs >= thr).astype(int)
    out = {
        'accuracy': accuracy_score(y_true, preds),
        'precision': precision_score(y_true, preds, zero_division=0),
        'recall': recall_score(y_true, preds, zero_division=0),
        'f1': f1_score(y_true, preds, zero_division=0),
    }
    try:
        out['roc_auc'] = roc_auc_score(y_true, probs)
    except Exception:
        out['roc_auc'] = None
    return out


def main(args):
    train_path = Path(args.train)
    test_path = Path(args.test) if args.test is not None else None

    if not train_path.exists():
        print(f"Train file {train_path} not found", file=sys.stderr)
        sys.exit(2)

    train = pd.read_csv(train_path)
    if test_path is not None and test_path.exists():
        test = pd.read_csv(test_path)
    else:
        test = None

    required = {'id', 'topic', 'answer'}
    if not required.issubset(train.columns):
        raise ValueError('train file must contain columns: id, topic, answer, is_cheating')
    if 'is_cheating' not in train.columns:
        raise ValueError('train file must contain is_cheating column')

    train['text'] = combine_text(train)
    if test is not None and {'topic', 'answer'}.issubset(test.columns):
        test['text'] = combine_text(test)

    # If test has labels, use it. Otherwise split train.
    if test is None or 'is_cheating' not in test.columns:
        train_df, val_df = train_test_split(train, test_size=0.2, stratify=train['is_cheating'], random_state=42)
        print('Using internal validation split')
    else:
        train_df = train
        val_df = test
        print('Using provided test file as validation')

    X_train_text = train_df['text'].tolist()
    y_train = train_df['is_cheating'].values
    X_val_text = val_df['text'].tolist()
    y_val = val_df['is_cheating'].values if 'is_cheating' in val_df.columns else None

    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)

    # TF-IDF branch
    print('Fitting TF-IDF vectorizer...')
    tfidf = TfidfVectorizer(ngram_range=(1,2), max_features=50000, min_df=2)
    X_train_tfidf = tfidf.fit_transform(X_train_text)
    X_val_tfidf = tfidf.transform(X_val_text)
    print('TF-IDF shapes:', X_train_tfidf.shape, X_val_tfidf.shape)

    clf_lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    clf_rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    clf_gb = GradientBoostingClassifier(n_estimators=100, random_state=42)

    print('Training TF-IDF classifiers...')
    clf_lr.fit(X_train_tfidf, y_train)
    clf_rf.fit(X_train_tfidf, y_train)
    clf_gb.fit(X_train_tfidf, y_train)

    voting_tfidf = VotingClassifier(estimators=[('lr', clf_lr), ('rf', clf_rf), ('gb', clf_gb)], voting='soft')
    voting_tfidf.fit(X_train_tfidf, y_train)
    val_probs_tfidf = voting_tfidf.predict_proba(X_val_tfidf)[:, 1]

    # Embedding branch (optional)
    val_probs_emb = None
    voting_emb = None
    if not args.no_embeddings:
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as e:
            print('sentence-transformers not installed or import failed:', e, file=sys.stderr)
            print('Run with --no-embeddings to skip embedding branch')
            raise

        print('Loading SentenceTransformer model (this may download weights)...')
        embedder = SentenceTransformer(args.embedder)

        def encode_texts(texts, batch_size=64):
            embeddings = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                emb = embedder.encode(batch, show_progress_bar=False, convert_to_numpy=True)
                embeddings.append(emb)
            return np.vstack(embeddings)

        print('Encoding train texts...')
        X_train_emb = encode_texts(X_train_text)
        print('Encoding val texts...')
        X_val_emb = encode_texts(X_val_text)
        print('Embeddings shapes:', X_train_emb.shape, X_val_emb.shape)

        clf_lr_emb = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
        clf_rf_emb = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
        clf_gb_emb = GradientBoostingClassifier(n_estimators=100, random_state=42)

        print('Training embedding classifiers...')
        clf_lr_emb.fit(X_train_emb, y_train)
        clf_rf_emb.fit(X_train_emb, y_train)
        clf_gb_emb.fit(X_train_emb, y_train)

        voting_emb = VotingClassifier(estimators=[('lr', clf_lr_emb), ('rf', clf_rf_emb), ('gb', clf_gb_emb)], voting='soft')
        voting_emb.fit(X_train_emb, y_train)
        val_probs_emb = voting_emb.predict_proba(X_val_emb)[:, 1]

    # Meta stacking
    if val_probs_emb is None:
        print('Only TF-IDF branch available; skipping stacking meta model')
        meta_clf = None
        meta_val_probs = val_probs_tfidf
    else:
        X_meta_train = np.vstack([voting_tfidf.predict_proba(X_train_tfidf)[:, 1], voting_emb.predict_proba(X_train_emb)[:, 1]]).T
        X_meta_val = np.vstack([val_probs_tfidf, val_probs_emb]).T
        meta_clf = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
        meta_clf.fit(X_meta_train, y_train)
        meta_val_probs = meta_clf.predict_proba(X_meta_val)[:, 1]

    # Evaluate
    print('\nEvaluation results:')
    if y_val is None:
        print('No labels available in validation set; skipping evaluation')
    else:
        results = {
            'tfidf': evaluate(y_val, val_probs_tfidf),
            'meta': evaluate(y_val, meta_val_probs)
        }
        if val_probs_emb is not None:
            results['embeddings'] = evaluate(y_val, val_probs_emb)
        print(results)
        print('\nClassification report (meta):')
        print(classification_report(y_val, (meta_val_probs >= 0.5).astype(int), zero_division=0))

    # Save artifacts
    joblib.dump(tfidf, models_dir / 'tfidf_vectorizer.joblib')
    joblib.dump(voting_tfidf, models_dir / 'voting_tfidf.joblib')
    if voting_emb is not None:
        joblib.dump(voting_emb, models_dir / 'voting_emb.joblib')
        # save embedder via its save method
        embedder.save(str(models_dir / 'sentence_transformer'))
    if meta_clf is not None:
        joblib.dump(meta_clf, models_dir / 'meta_clf.joblib')

    print('Saved models to', models_dir)

    # Create a submission file (id, prediction) if a test file exists
    # Prediction is binary 0/1 by default; use --output-prob to save probabilities instead
    test_path = Path(args.test) if args.test is not None else None
    if test_path is not None and test_path.exists():
        test_df = pd.read_csv(test_path)
        if {'id', 'topic', 'answer'}.issubset(test_df.columns):
            test_df['text'] = combine_text(test_df)
            X_test_text = test_df['text'].tolist()
            # TF-IDF predictions
            X_test_tfidf = tfidf.transform(X_test_text)
            p_tfidf = voting_tfidf.predict_proba(X_test_tfidf)[:, 1]

            final_prob = p_tfidf
            # If embedding branch exists, encode and combine
            if 'voting_emb' in locals() and voting_emb is not None:
                # encode texts in batches using the same approach as training
                def _encode_texts_local(texts, batch_size=64):
                    embeddings = []
                    for i in range(0, len(texts), batch_size):
                        batch = texts[i:i+batch_size]
                        emb = embedder.encode(batch, show_progress_bar=False, convert_to_numpy=True)
                        embeddings.append(emb)
                    return np.vstack(embeddings)

                X_test_emb = _encode_texts_local(X_test_text)
                p_emb = voting_emb.predict_proba(X_test_emb)[:, 1]
                if meta_clf is not None:
                    X_meta_test = np.vstack([p_tfidf, p_emb]).T
                    final_prob = meta_clf.predict_proba(X_meta_test)[:, 1]
                else:
                    # fallback to TF-IDF probabilities if meta missing
                    final_prob = p_tfidf

            # Choose output form
            if getattr(args, 'output_prob', False):
                pred_col = final_prob
            else:
                pred_col = (final_prob >= 0.5).astype(int)

            submission = pd.DataFrame({'id': test_df['id'], 'prediction': pred_col})
            submission_path = models_dir / 'submission.csv'
            submission.to_csv(submission_path, index=False)
            print('Saved submission to', submission_path)
        else:
            print('Test file missing required columns for submission (id, topic, answer)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', required=False, default='traindata.csv', help='Path to training csv (default: traindata.csv)')
    parser.add_argument('--test', required=False, default='testdata.csv', help='Path to test csv (default: testdata.csv)')
    parser.add_argument('--output-prob', action='store_true', help='Write probabilities in submission instead of binary predictions')
    parser.add_argument('--no-embeddings', action='store_true', help='Skip transformer embeddings (faster)')
    parser.add_argument('--embedder', default='all-MiniLM-L6-v2', help='SentenceTransformer model name')
    args = parser.parse_args()
    main(args)

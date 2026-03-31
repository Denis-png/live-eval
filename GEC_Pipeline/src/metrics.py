# src/metrics.py
import errant
import numpy as np
from nltk.translate.gleu_score import sentence_gleu

# Initialize the ERRANT annotator globally within the module
try:
    annotator = errant.load("en")
except Exception:
    # Fallback if spacy model isn't linked
    import spacy
    nlp = spacy.load("en_core_web_sm")
    annotator = errant.load("en", nlp)

def calculate_gleu(original_sentences, predicted_sentences):
    """
    Computes the average GLEU score for a set of predictions.
    Higher is better (fluency).
    """
    scores = [
        sentence_gleu([orig.split()], pred.split())
        for orig, pred in zip(original_sentences, predicted_sentences)
    ]
    return round(float(np.mean(scores)), 4)

def calculate_errant(corrupted_sentences, original_sentences, predicted_sentences):
    """
    Computes precision, recall, and F0.5 using the ERRANT framework.
    """
    tp, fp, fn = 0, 0, 0
    
    for corrupted, original, prediction in zip(corrupted_sentences, original_sentences, predicted_sentences):
        # Parse sentences into ERRANT documents
        orig_doc = annotator.parse(corrupted)
        ref_doc = annotator.parse(original)
        pred_doc = annotator.parse(prediction)

        # Extract edits
        ref_edits = annotator.annotate(orig_doc, ref_doc)
        pred_edits = annotator.annotate(orig_doc, pred_doc)

        # Convert edits to sets for comparison (start, end, type)
        ref_set = {(e.o_start, e.o_end, e.type) for e in ref_edits}
        pred_set = {(e.o_start, e.o_end, e.type) for e in pred_edits}

        tp += len(pred_set & ref_set)
        fp += len(pred_set - ref_set)
        fn += len(ref_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # F0.5 weights precision twice as much as recall
    f05 = (1.25 * precision * recall) / (0.25 * precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f0.5": round(f05, 4)
    }
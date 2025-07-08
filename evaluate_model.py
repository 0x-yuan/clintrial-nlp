#!/usr/bin/env python3

import json
import os
import sys
import warnings
from sklearn.metrics import f1_score, precision_score, recall_score

warnings.simplefilter('ignore')

# ============================================================================
# EVALUATION FUNCTIONS (from evaluate.py)
# ============================================================================

def extract_control_set(predictions, gold):
    control_predictions = {}
    for key in gold.keys():
        if "Causal_type" not in gold[key].keys():
            control_predictions[key] = predictions[key]
    return control_predictions

def extract_by_intervention(predictions, gold):
    para_predictions = {}
    cont_predictions = {}
    numerical_para_predictions = {}
    numerical_cont_predictions = {}
    definitions_predictions = {}
    for key in predictions.keys():
        if "Intervention" not in gold[key].keys():
            continue
        if gold[key]["Intervention"] == "Paraphrase":
            para_predictions[key] = predictions[key]
        elif gold[key]["Intervention"] == "Contradiction":
            cont_predictions[key] = predictions[key]
        elif gold[key]["Intervention"] == "Numerical_paraphrase":
            numerical_para_predictions[key] = predictions[key]
        elif gold[key]["Intervention"] == "Numerical_contradiction":
            numerical_cont_predictions[key] = predictions[key]
        elif gold[key]["Intervention"] == "Text_appended":
            definitions_predictions[key] = predictions[key]
    return para_predictions, cont_predictions, numerical_para_predictions, numerical_cont_predictions, definitions_predictions

def extract_by_causal_type(predictions, gold):
    predictions_preserving = {}
    predictions_altering = {}
    for key in predictions.keys():
        if "Causal_type" not in gold[key].keys():
            continue
        if gold[key]["Causal_type"][0] == "Preserving":
            predictions_preserving[key] = predictions[key]
        elif gold[key]["Causal_type"][0] == "Altering":
            predictions_altering[key] = predictions[key]
    return predictions_preserving, predictions_altering

def faithfulness(predictions, gold):
    uuid_list = list(predictions.keys())
    N = len(uuid_list)
    results = []
    for key in uuid_list:
        if predictions[key]["Prediction"] != gold[gold[key]["Causal_type"][1]]["Label"]:
            results.append(1)
        else:
            results.append(0)
    Faithfulness = sum(results) / N
    return Faithfulness

def consistency(predictions_preserving, predictions, gold):
    uuid_list = list(predictions_preserving.keys())
    N = len(uuid_list)
    results = []
    for key in uuid_list:
        if predictions_preserving[key]["Prediction"] == predictions[gold[key]["Causal_type"][1]]["Prediction"]:
            results.append(1)
        else:
            results.append(0)
    Consistency = sum(results) / N
    return Consistency

def extract_contrast_set(predictions, gold):
    contrast_predictions = {}
    for key in predictions.keys():
        if "Causal_type" in gold[key].keys():
            contrast_predictions[key] = predictions[key]
    return contrast_predictions

def F1_Recall_Precision(predictions, gold):
    pred_labels = []
    gold_labels = []
    for key in predictions.keys():
        if predictions[key]["Prediction"] == "Entailment":
            pred_labels.append(1)
        else:
            pred_labels.append(0)
        if gold[key]["Label"] == "Entailment":
            gold_labels.append(1)
        else:
            gold_labels.append(0)
    F1 = f1_score(gold_labels, pred_labels)
    Recall = precision_score(gold_labels, pred_labels)
    Precision = recall_score(gold_labels, pred_labels)
    return F1, Recall, Precision

# ============================================================================
# VALIDATION AND EVALUATION FUNCTIONS
# ============================================================================

def validate_predictions(predictions_file, test_file):
    """Validate the predictions file format and completeness"""
    
    # Load test data to get expected UUIDs
    try:
        with open(test_file, 'r') as f:
            test_data = json.load(f)
        print(f"✅ Loaded {len(test_data)} test entries")
    except Exception as e:
        print(f"❌ Error loading test.json: {e}")
        return False, None, None
    
    # Load predictions
    try:
        with open(predictions_file, 'r') as f:
            predictions = json.load(f)
        print(f"✅ Loaded {len(predictions)} predictions")
    except Exception as e:
        print(f"❌ Error loading predictions: {e}")
        return False, None, None
    
    # Check completeness
    test_uuids = set(test_data.keys())
    pred_uuids = set(predictions.keys())
    
    missing = test_uuids - pred_uuids
    extra = pred_uuids - test_uuids
    
    if missing:
        print(f"❌ Missing {len(missing)} predictions for test UUIDs")
        if len(missing) <= 10:
            print(f"   Missing: {list(missing)}")
        return False, None, None
    
    if extra:
        print(f"⚠️  Found {len(extra)} extra predictions (not in test data)")
        if len(extra) <= 10:
            print(f"   Extra: {list(extra)}")
    
    # Validate prediction format
    valid_labels = {"Entailment", "Contradiction"}
    invalid_count = 0
    
    for uuid, pred in predictions.items():
        if not isinstance(pred, dict):
            print(f"❌ Invalid format for {uuid}: expected dict")
            invalid_count += 1
        elif "Prediction" not in pred:
            print(f"❌ Missing 'Prediction' field for {uuid}")
            invalid_count += 1
        elif pred["Prediction"] not in valid_labels:
            print(f"❌ Invalid label '{pred['Prediction']}' for {uuid}")
            invalid_count += 1
    
    if invalid_count > 0:
        print(f"❌ Found {invalid_count} invalid predictions")
        return False, None, None
    
    print("✅ All predictions validated successfully!")
    return True, predictions, test_data

def run_evaluation(predictions, gold):
    """Run the complete evaluation and return results"""
    
    results = {}
    
    # Control Test Set F1, Recall, Precision PUBLIC
    control_predictions = extract_control_set(predictions, gold)
    Control_F1, Control_Rec, Control_Prec = F1_Recall_Precision(control_predictions, gold)
    results.update({
        'Control_F1': Control_F1,
        'Control_Recall': Control_Rec,
        'Control_Precision': Control_Prec
    })

    # Contrast Consistency & Faithfulness PUBLIC
    contrast_predictions = extract_contrast_set(predictions, gold)
    predictions_preserving, predictions_altering = extract_by_causal_type(contrast_predictions, gold)
    Faithfulness = faithfulness(predictions_altering, gold)
    Consistency = consistency(predictions_preserving, predictions, gold)
    results.update({
        'Contrast_F1': F1_Recall_Precision(contrast_predictions, gold)[0],
        'Contrast_Recall': F1_Recall_Precision(contrast_predictions, gold)[1],
        'Contrast_Precision': F1_Recall_Precision(contrast_predictions, gold)[2],
        'Faithfulness': Faithfulness,
        'Consistency': Consistency
    })

    # Intervention-wise Consistency & Faithfulness HIDDEN
    para_predictions, cont_predictions, numerical_para_predictions, numerical_cont_predictions, definitions_predictions = \
        extract_by_intervention(predictions, gold)
    
    para_preserving = extract_by_causal_type(para_predictions, gold)[0]
    cont_preserving, cont_altering = extract_by_causal_type(cont_predictions, gold)
    numerical_para_preserving = extract_by_causal_type(numerical_para_predictions, gold)[0]
    numerical_cont_preserving, numerical_cont_altering = extract_by_causal_type(numerical_cont_predictions, gold)
    definitions_preserving = extract_by_causal_type(definitions_predictions, gold)[0]
    
    para_Consistency = consistency(para_preserving, predictions, gold)
    cont_Faithfulness = faithfulness(cont_altering, gold)
    cont_Consistency = consistency(cont_preserving, predictions, gold)
    numerical_para_Consistency = consistency(numerical_para_preserving, predictions, gold)
    numerical_cont_Faithfulness = faithfulness(numerical_cont_altering, gold)
    numerical_cont_Consistency = consistency(numerical_cont_preserving, predictions, gold)
    definitions_Consistency = consistency(definitions_preserving, predictions, gold)
    
    results.update({
        'Para_Consistency': para_Consistency,
        'Cont_Faithfulness': cont_Faithfulness,
        'Cont_Consistency': cont_Consistency,
        'Numerical_Para_Consistency': numerical_para_Consistency,
        'Numerical_Cont_Faithfulness': numerical_cont_Faithfulness,
        'Numerical_Cont_Consistency': numerical_cont_Consistency,
        'Definitions_Consistency': definitions_Consistency
    })

    # Intervention-wise F1, Recall, Precision HIDDEN
    para_F1, para_Rec, para_Prec = F1_Recall_Precision(para_predictions, gold)
    cont_F1, cont_Rec, cont_Prec = F1_Recall_Precision(cont_predictions, gold)
    numerical_para_F1, numerical_para_Rec, numerical_para_Prec = F1_Recall_Precision(numerical_para_predictions, gold)
    numerical_cont_F1, numerical_cont_Rec, numerical_cont_Prec = F1_Recall_Precision(numerical_cont_predictions, gold)
    definitions_F1, definitions_Rec, definitions_Prec = F1_Recall_Precision(definitions_predictions, gold)
    
    results.update({
        'Para_F1': para_F1,
        'Para_Recall': para_Rec,
        'Para_Precision': para_Prec,
        'Cont_F1': cont_F1,
        'Cont_Recall': cont_Rec,
        'Cont_Precision': cont_Prec,
        'Numerical_Para_F1': numerical_para_F1,
        'Numerical_Para_Recall': numerical_para_Rec,
        'Numerical_Para_Precision': numerical_para_Prec,
        'Numerical_Cont_F1': numerical_cont_F1,
        'Numerical_Cont_Recall': numerical_cont_Rec,
        'Numerical_Cont_Precision': numerical_cont_Prec,
        'Definitions_F1': definitions_F1,
        'Definitions_Recall': definitions_Rec,
        'Definitions_Precision': definitions_Prec
    })
    
    return results

def display_results(results):
    """Display the evaluation results in a formatted way"""
    
    print(f"\n📊 Evaluation Results:")
    print("=" * 60)
    
    # Display main metrics
    print("\n🎯 Main Metrics:")
    main_metrics = [
        "Control_F1", "Control_Recall", "Control_Precision",
        "Contrast_F1", "Contrast_Recall", "Contrast_Precision",
        "Faithfulness", "Consistency"
    ]
    
    for metric in main_metrics:
        if metric in results:
            print(f"  {metric:<20}: {results[metric]:.4f}")
    
    # Display intervention-specific metrics
    print("\n🔍 Intervention-Specific Metrics:")
    intervention_metrics = [
        "Para_Consistency", "Cont_Faithfulness", "Cont_Consistency",
        "Numerical_Para_Consistency", "Numerical_Cont_Faithfulness",
        "Numerical_Cont_Consistency", "Definitions_Consistency"
    ]
    
    for metric in intervention_metrics:
        if metric in results:
            print(f"  {metric:<30}: {results[metric]:.4f}")
    
    # Display F1 scores for different interventions
    print("\n📈 F1 Scores by Intervention:")
    f1_metrics = [
        "Para_F1", "Cont_F1", "Numerical_Para_F1", 
        "Numerical_Cont_F1", "Definitions_F1"
    ]
    
    for metric in f1_metrics:
        if metric in results:
            print(f"  {metric:<20}: {results[metric]:.4f}")
    
    # Performance summary
    print(f"\n🏆 Performance Summary:")
    main_score = (results.get("Control_F1", 0) + 
                 results.get("Faithfulness", 0) + 
                 results.get("Consistency", 0)) / 3
    print(f"  Average Main Score: {main_score:.4f}")
    
    if main_score > 0.70:
        print("  🎉 Excellent performance!")
    elif main_score > 0.60:
        print("  👍 Good performance!")
    elif main_score > 0.50:
        print("  🔧 Room for improvement")
    else:
        print("  📚 Consider reviewing your approach")

def main():
    if len(sys.argv) != 2:
        print("Usage: python evaluate_model.py <predictions_file>")
        print("Example: python evaluate_model.py my_predictions.json")
        sys.exit(1)
    
    predictions_file = sys.argv[1]
    
    print(f"🚀 Starting evaluation of {predictions_file}")
    print("=" * 50)
    
    # Validate input files
    if not os.path.exists(predictions_file):
        print(f"❌ Predictions file not found: {predictions_file}")
        sys.exit(1)
    
    if not os.path.exists("test.json"):
        print("❌ test.json not found in current directory")
        sys.exit(1)
    
    if not os.path.exists("gold_test.json"):
        print("❌ gold_test.json not found in current directory")
        sys.exit(1)
    
    # Load and validate predictions
    valid, predictions, test_data = validate_predictions(predictions_file, "test.json")
    if not valid:
        print("\n❌ Validation failed. Please check the errors above.")
        sys.exit(1)
    
    # Load gold standard
    try:
        with open("gold_test.json", 'r') as f:
            gold = json.load(f)
        print(f"✅ Loaded gold standard with {len(gold)} entries")
    except Exception as e:
        print(f"❌ Error loading gold_test.json: {e}")
        sys.exit(1)
    
    # Run evaluation
    print(f"\n🔄 Running evaluation...")
    try:
        results = run_evaluation(predictions, gold)
        print("✅ Evaluation completed successfully!")
        
        # Display results
        display_results(results)
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 
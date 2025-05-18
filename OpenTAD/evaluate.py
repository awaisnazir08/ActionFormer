import json
import numpy as np
import argparse
from collections import defaultdict


def calculate_IoU(pred_segment, gt_segment):
    """
    Calculate Intersection over Union between two segments
    """
    pred_start, pred_end = pred_segment
    gt_start, gt_end = gt_segment
    
    # Calculate intersection
    intersection_start = max(pred_start, gt_start)
    intersection_end = min(pred_end, gt_end)
    
    if intersection_end <= intersection_start:
        return 0.0  # No intersection
    
    # Calculate areas
    intersection = intersection_end - intersection_start
    union = (pred_end - pred_start) + (gt_end - gt_start) - intersection
    
    return float(intersection) / union


def calculate_accuracy(results, ground_truth):
    """
    Calculate frame-level and segment-level accuracy
    
    Frame-level: What percentage of the video timeline is correctly classified
    Segment-level: What percentage of segments are correctly detected
    """
    # For frame-level accuracy, we'll discretize the timeline into small intervals
    frame_interval = 0.05  # 50ms intervals
    
    total_frames = 0
    correct_frames = 0
    
    total_segments = 0
    correct_segments = 0
    
    for video_id in results:
        if video_id not in ground_truth:
            continue
            
        # Get video duration
        video_duration = ground_truth[video_id].get("duration", 0)
        if video_duration == 0:
            # Try to find maximum segment end time as duration
            for annot in ground_truth[video_id]["annotations"]:
                video_duration = max(video_duration, annot["segment"][1])
        
        # Convert predictions and ground truth to frame-level labels
        num_frames = int(video_duration / frame_interval) + 1
        total_frames += num_frames
        
        # Initialize arrays for predictions and ground truth
        pred_frames = ["background"] * num_frames
        gt_frames = ["background"] * num_frames
        
        # Fill in ground truth frames
        for annot in ground_truth[video_id]["annotations"]:
            start_frame = int(annot["segment"][0] / frame_interval)
            end_frame = int(annot["segment"][1] / frame_interval)
            for i in range(start_frame, min(end_frame + 1, num_frames)):
                gt_frames[i] = annot["label"]
        
        # Fill in prediction frames
        for pred in results[video_id]:
            start_frame = int(pred["segment"][0] / frame_interval)
            end_frame = int(pred["segment"][1] / frame_interval)
            for i in range(start_frame, min(end_frame + 1, num_frames)):
                pred_frames[i] = pred["label"]
        
        # Count correct frames
        for i in range(num_frames):
            if pred_frames[i] == gt_frames[i]:
                correct_frames += 1
        
        # For segment-level accuracy, we'll use IoU > 0.5 as correct detection
        gt_segments = ground_truth[video_id]["annotations"]
        total_segments += len(gt_segments)
        
        # Track which ground truth segments have been correctly detected
        gt_detected = [False] * len(gt_segments)
        
        for pred in results[video_id]:
            pred_segment = pred["segment"]
            pred_label = pred["label"]
            
            for gt_idx, gt in enumerate(gt_segments):
                gt_segment = gt["segment"]
                gt_label = gt["label"]
                
                if pred_label == gt_label:
                    iou = calculate_IoU(pred_segment, gt_segment)
                    if iou > 0.5:  # Using 0.5 IoU threshold for segment accuracy
                        gt_detected[gt_idx] = True
                        break
        
        correct_segments += sum(gt_detected)
    
    # Calculate accuracies
    frame_accuracy = correct_frames / total_frames if total_frames > 0 else 0
    segment_accuracy = correct_segments / total_segments if total_segments > 0 else 0
    
    return frame_accuracy, segment_accuracy

def calculate_f1_at_threshold(results, ground_truth, iou_threshold):
    """
    Calculate F1 score at specified IoU threshold
    """
    total_gt = 0
    total_pred = 0
    total_correct = 0
    
    for video_id in results:
        if video_id not in ground_truth:
            print(f"Warning: Video {video_id} in results but not in ground truth.")
            continue
        
        # Get predictions and ground truth for this video
        video_preds = results[video_id]
        video_gt = ground_truth[video_id]["annotations"]
        
        # Count total ground truth segments
        total_gt += len(video_gt)
        total_pred += len(video_preds)
        
        # Track which ground truth segments have been matched
        gt_matched = [False] * len(video_gt)
        
        # For each prediction, find the best matching ground truth
        for pred in video_preds:
            pred_segment = pred["segment"]
            pred_label = pred["label"]
            
            best_iou = 0
            best_gt_idx = -1
            
            # Find the best matching ground truth for this prediction
            for gt_idx, gt in enumerate(video_gt):
                if gt_matched[gt_idx]:
                    continue  # Skip already matched ground truths
                
                gt_segment = gt["segment"]
                gt_label = gt["label"]
                
                # Only consider if labels match
                if pred_label == gt_label:
                    iou = calculate_IoU(pred_segment, gt_segment)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
            
            # If we found a match above the threshold
            if best_iou >= iou_threshold and best_gt_idx >= 0:
                gt_matched[best_gt_idx] = True
                total_correct += 1
    
    # Calculate precision, recall, and F1
    precision = total_correct / total_pred if total_pred > 0 else 0
    recall = total_correct / total_gt if total_gt > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1


def main():
    parser = argparse.ArgumentParser(description='Calculate action detection metrics')
    parser.add_argument('--results', required=True, help='Path to results.json file')
    parser.add_argument('--labels', required=True, help='Path to labels.json file')
    parser.add_argument('--score_threshold', type=float, default=0.3, 
                        help='Minimum confidence score threshold (default: 0.3)')
    args = parser.parse_args()
    
    # Load results and ground truth data
    with open(args.results, 'r') as f:
        results_data = json.load(f)
    
    with open(args.labels, 'r') as f:
        gt_data = json.load(f)
    
    # Process results data into a more accessible format
    results = {}
    if "results" in results_data:
        results_data = results_data["results"]
    
    # Filter predictions by score threshold
    filtered_count = 0
    total_count = 0
    
    for video_id, video_results in results_data.items():
        # Filter out predictions with score less than threshold
        filtered_results = []
        for pred in video_results:
            total_count += 1
            if "score" in pred and pred["score"] >= args.score_threshold:
                filtered_results.append(pred)
            else:
                filtered_count += 1
        
        results[video_id] = filtered_results
    
    # Process ground truth data
    ground_truth = gt_data["database"]
    
    # Print filtering summary
    print(f"Filtering Results:")
    print(f"Total predictions: {total_count}")
    print(f"Filtered out (score < {args.score_threshold}): {filtered_count}")
    print(f"Remaining predictions: {total_count - filtered_count}")
    
    # Calculate accuracy
    frame_accuracy, segment_accuracy = calculate_accuracy(results, ground_truth)
    print(f"\nAccuracy Metrics:")
    print(f"Frame-level Accuracy: {frame_accuracy*100:.4f}%")
    print(f"Segment-level Accuracy: {segment_accuracy*100:.4f}%")
    
    # Calculate metrics at different IoU thresholds
    iou_thresholds = [0.1, 0.25, 0.5]
    
    print("\nF1 Score Metrics:")
    print("-" * 50)
    
    for threshold in iou_thresholds:
        precision, recall, f1 = calculate_f1_at_threshold(results, ground_truth, threshold)
        print(f"F1@{threshold*100:.0f}%: {f1*100:.4f}")
        print(f"Precision@{threshold*100:.0f}%: {precision*100:.4f}")
        print(f"Recall@{threshold*100:.0f}%: {recall*100:.4f}")
        print("-" * 50)


if __name__ == "__main__":
    main()
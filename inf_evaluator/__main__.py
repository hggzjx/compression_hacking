import argparse
from .evaluator import InfEvaluator

def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM with refined informatics metrics")

    # experimental settings
    parser.add_argument("--model_path",required=True, type=str, default="models/opt-1.3b")
    parser.add_argument("--init_model_path", type=str, default=None)
    
    # parser.add_argument("--dataset_name", type=str, default="truthfulqa")
    parser.add_argument("--sample_size", type=int, default=800)
    parser.add_argument("--metric", type=str, default="semantic_cv")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--visible_devices", type=str, default="0,1,2")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--record_path", type=str, default="experiment_records")

    #metric lab settings
    parser.add_argument("--feature_clipped", action="store_true")
    parser.add_argument("--K_aniso", type=int, default=1)
    parser.add_argument("--razor_type", type=str, default='pcs')
    parser.add_argument("--K_whitening", type=int, default=16)
    parser.add_argument("--K_remove_direction", type=int, default=1)

    args = parser.parse_args()

    evaluator_kwargs = {"model_path": args.model_path,
                        "sample_size": args.sample_size,
                        "batch_size": args.batch_size,
                        "visible_devices": args.visible_devices,
                        "debug": args.debug, 
                        "init_model_path": args.init_model_path,
                        "record_path": args.record_path}   
    
    evaluate_kwargs = {"metric": args.metric,
                       "feature_clipped": args.feature_clipped,
                       "K_aniso": args.K_aniso,
                       "razor_type": args.razor_type,
                       "K_whitening": args.K_whitening,
                       "K_remove_direction": args.K_remove_direction}
    
    evaluator = InfEvaluator(**evaluator_kwargs)
    evaluator.evaluate(**evaluate_kwargs)

if __name__ == "__main__":
    main()
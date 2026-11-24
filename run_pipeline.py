"""Train and evaluate the full MediGuard pipeline and write artifacts to ./artifacts.

    python run_pipeline.py            # full run (~6 min on a laptop)
    python run_pipeline.py --quick    # one configuration per model family
"""
from mediguard.training.train import main

if __name__ == "__main__":
    main()

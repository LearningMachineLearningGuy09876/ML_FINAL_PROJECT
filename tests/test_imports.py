import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")

if SRC not in sys.path:
    sys.path.append(SRC)


def test_imports():
    import config
    import data_loader
    import recommender
    import user_profiles
    import search_engine
    import chatbot
    import metrics
    import poster_utils
    import utils

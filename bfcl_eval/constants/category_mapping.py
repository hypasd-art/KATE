VERSION_PREFIX = "BFCL_v4"


ALL_AVAILABLE_MEMORY_BACKENDS = [
    "kv",
    "vector",
    "rec_sum",
]

NON_LIVE_CATEGORY = [
    "simple_python",
    "simple_java",
    "simple_javascript",
    "multiple",
    "parallel",
    "parallel_multiple",
    "irrelevance",
    # "exec_simple",
    # "exec_parallel",
    # "exec_multiple",
    # "exec_parallel_multiple",
    # "rest",
    # "sql",
    # "chatable",
]
LIVE_CATEGORY = [
    "live_simple",
    "live_multiple",
    "live_parallel",
    "live_parallel_multiple",
    "live_irrelevance",
    "live_relevance",
]
MULTI_TURN_CATEGORY = [
    "multi_turn_base",
    "compute",
    "multi_turn_base_training",
    "multi_turn_base_testing",
    "multi_turn_miss_func",
    "multi_turn_miss_func_training",
    "multi_turn_miss_func_testing",
    "multi_turn_miss_param",
    "multi_turn_miss_param_training",
    "multi_turn_miss_param_testing",
    "multi_turn_long_context",
    "multi_turn_long_context_training",
    "multi_turn_long_context_testing",
    "multi_turn_training_data"
    # "multi_turn_composite",
]
WEB_SEARCH_CATEGORY = [
    "web_search_base",
    "web_search_no_snippet",
    "web_search_training",
    "web_search_testing",
    "web_search_no_snippet_training",
    "web_search_no_snippet_testing",
]

MEMORY_CATEGORY = [f"memory_{backend}" for backend in ALL_AVAILABLE_MEMORY_BACKENDS] #  + 
MEMORY_CATEGORY_TRAINING = [f"memory_{backend}_training" for backend in ALL_AVAILABLE_MEMORY_BACKENDS] # + 
MEMORY_CATEGORY_TESTING = [f"memory_{backend}_testing" for backend in ALL_AVAILABLE_MEMORY_BACKENDS]
MEMORY_SCENARIO_NAME = [
    "student",
    "customer",
    "finance",
    "healthcare",
    "notetaker",
]


SINGLE_TURN_CATEGORY = NON_LIVE_CATEGORY + LIVE_CATEGORY
AGENTIC_CATEGORY = MEMORY_CATEGORY + WEB_SEARCH_CATEGORY
NON_SCORING_CATEGORY = ["format_sensitivity"]

ALL_SCORING_CATEGORIES = SINGLE_TURN_CATEGORY + MULTI_TURN_CATEGORY + AGENTIC_CATEGORY
ALL_CATEGORIES = ALL_SCORING_CATEGORIES + NON_SCORING_CATEGORY

TEST_COLLECTION_MAPPING = {
    "all": ALL_CATEGORIES,
    "all_scoring": ALL_SCORING_CATEGORIES,
    "multi_turn": MULTI_TURN_CATEGORY,
    "single_turn": SINGLE_TURN_CATEGORY,
    "live": LIVE_CATEGORY,
    "non_live": NON_LIVE_CATEGORY,
    "non_python": [
        "simple_java",
        "simple_javascript",
    ],
    "python": [
        "simple_python",
        "irrelevance",
        "parallel",
        "multiple",
        "parallel_multiple",
        "live_simple",
        "live_multiple",
        "live_parallel",
        "live_parallel_multiple",
        "live_irrelevance",
        "live_relevance",
    ],
    "memory": MEMORY_CATEGORY,
    "memory_training": MEMORY_CATEGORY_TRAINING,
    "memory_testing": MEMORY_CATEGORY_TESTING,
    "web_search": WEB_SEARCH_CATEGORY,
    "agentic": AGENTIC_CATEGORY,
}

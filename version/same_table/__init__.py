"""Same Table / 糖友同枱 — core package.

Layers:
    library    curated dish rows (the only source of dishes)
    tagging    Pipeline 1: understand tonight's request  -> tags
    retrieval  Pipeline 2: rank library dishes for the request
    rules      plain-Python rules: dish counts, category balance, 14-day log,
               20-minute filter, patient actions, shopping-list merge
    planner    glue: text + chips + headcount -> menu plan
"""

__version__ = "1.0.0"

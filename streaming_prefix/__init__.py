"""
streaming_prefix: Innovation 3 — learned recursive state-space update for
VLM prefix encoding, replacing full re-encoding at every control step.

Phase A (this package's first stage): pure compute profiling. Measures
what fraction of per-control-step inference latency is spent re-encoding
the (nearly-unchanged) visual prefix vs. running the flow-matching action
head, on the ACTUAL SimVLA architecture. This does not require a trained
checkpoint -- timing depends on compute graph shape, not weight values --
so it can run before any training and determines whether the streaming
state-update design (Phase B/C) is worth building at all.
"""

from parameter import Parameter
import copy

''' recursively populate the posteriors of each parameter in the event dictionary from the trace object'''
def trace_to_event(trace,event):
    if isinstance(event, dict):
        for key, value in event.items():
            if isinstance(value, dict):
                for d in trace_to_event(trace, value):
                    return d
            elif isinstance(value, list) or isinstance(value, tuple):
                try:
                    for v in value:
                        for d in trace_to_event(trace, v):
                            return d
                except:
                    pass
            elif isinstance(value, Parameter):
                value.posterior = trace.posterior[value.label]

    elif isinstance(value, Parameter):
        value.posterior = trace.posterior[value.label]


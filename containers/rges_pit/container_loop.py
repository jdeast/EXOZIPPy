"""The NENA task loop around component_pipeline.pipeline_function.

Locally (NENA_DEPLOYMENT_TYPE unset or 'local') this fits every JSON event in
./input_events and writes the output events to ./output_events.  In the
cloud it polls the component's SQS queue forever.

A failed task is reported with task.log_failure and emits no event; its SQS
message is not deleted, so it becomes visible again after the visibility
timeout (twice upper_bound_run_time__seconds) and is retried.
"""

import traceback

from component_pipeline import pipeline_function
from nena_component_tools.task_iterator import create_task_iterator


def main():
    for task in create_task_iterator():
        try:
            output_dictionary = pipeline_function(task.input_dictionary)
        except Exception as exc:
            task.log_failure(
                dictionary={
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            continue
        task.emit_event(output_dictionary=output_dictionary)


if __name__ == "__main__":
    main()

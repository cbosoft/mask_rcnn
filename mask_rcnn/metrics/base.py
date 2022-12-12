from typing import Dict


class MetricObject:
    """
    Object which calculates metrics for a batch of results.

    Results are collected across the batch before being finalised and returned.

    During validation, the methods should be called in order:

    ```
    ...
    metrics.batch_initialise()
    for batch in dataloader:
        ...
        # run model, get output
        ...
        metrics.batch_update(out, tgt)
    metric_values = metrics.batch_finalise()
    ```
    """

    def batch_initialise(self):
        raise NotImplementedError

    def batch_update(self, out, tgt):
        """
        `out` is the output from the mask r-cnn
        a list of dictionaries where each dictionary is the result from a single image
        Each dictionary has keys/values:
          - boxes  :: tensor of bboxes; shape (n, 4)
          - labels :: tensor indicating class; shape (n,)
          - scores :: tensor indicating confidence score; shape (n,)
          - masks  :: tensor of masks indicating position of object; shape (n, c, h, w)
        """
        raise NotImplementedError

    def batch_finalise(self) -> Dict[str, float]:
        raise NotImplementedError

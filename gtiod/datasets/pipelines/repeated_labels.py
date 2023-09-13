from mmdet.datasets.builder import PIPELINES
from mmcv.parallel import DataContainer as DC
from mmdet.datasets.pipelines.formatting import to_tensor

@PIPELINES.register_module()
class LoadRepeatedLabels:
    """Load repeated labels, similar to load proposal pipeline from original mmdetection.

    Required key is "annotator". Updated keys are "annotator".
    """

    def __call__(self, results):
        """Call function to load annotators from file.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded annotator/repeated label annotations.
        """
        results['coder'] = results['ann_info']['coder'].copy()
        return results

@PIPELINES.register_module()
class DefaultFormatBundleRepeatedLabels:
    """
        This makes the DefaultFormatBundle operation for the repeated labels, similar to the original packages operation.
    """
    def __call__(self, results):
        assert "coder" in results, "Key coder should be in {}".format(results)
        results["coder"] = DC( to_tensor(results["coder"]) )

        return results

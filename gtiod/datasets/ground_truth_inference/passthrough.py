from .builder import GTINFERENCE

@GTINFERENCE.register_module()
class PassthroughPre():
    def __init__(self, **kwargs):
        """ This is a mock function that can be used as an example to understand the worflow of implementing a custom
            ground truth inference function integrated into a repeated labels dataset.
            -> __init__ should take the dataset and the configuration from the config file. With the help of those,
               hyper-parameters should be set and preprocessing if necessary should be done.
        """
        print("Found the following parameters in config:")
        for key, val in kwargs.items():
            # Please don't print the parameters in the "real" method
            print("{} - {}".format(key, val))

    def __call__(self, ann_file):
        """
            This is executed when the dataset is constructed during __init__.

            You should use the https://docs.python.org/3/library/tempfile.html library to create a temporary pre-processed
            annotations file. This usually includes the following steps:
            1) open the annotation file
            2) do the processing steps depending on your set config parameter
            3) store the edited annotations files in a tempfile with the new name new_ann_file
            4) return the path to the new_ann_file so that during load_annotations the preprocessing is applied
        """
        new_ann_file = ann_file
        return new_ann_file


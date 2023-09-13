from .builder import ANNOTATION_PREPROCESSING

@ANNOTATION_PREPROCESSING.register_module()
class PassthroughMain():
    def __init__(self, dataset=None, **kwargs):
        """ This is a mock function that can be used as an example to understand the workflow of implementing a custom
            ground truth inference function integrated into a repeated labels dataset.
            -> __init__ should take the dataset and the configuration from the config file. With the help of those,
               hyper-parameters should be set and preprocessing if necessary should be done.
        """
        print("Found the following parameters in config:")
        for key, val in kwargs.items():
            # Please don't print the parameters in the "real" method
            print("{} - {}".format(key, val))

    def __call__(self, data_info, ann_info):
        """
            This is executed when the class is called in the get_ann_info function

            When the function is called it recieves information about the image in data_info and information about
            the annotation in ann_info. Both data elements can be changed and then handed back in the same order to
            the get_ann_info function
        """
        return data_info, ann_info
import sys
import numpy as np
from collections import defaultdict
from itertools import combinations, product
from copy import deepcopy

from .builder import GTINFERENCE
from .majority_voting import MajorityVoting

@GTINFERENCE.register_module()
class LocalizationAwareEM(MajorityVoting):
    def __init__(self, new_ann_path: str, annotator_key: str, iou_threshold: float, merging_ops: str, mask_threshold_ratio_factor: float =0.5, return_confusion_matrix: bool =False):
        """
        Dawid and Skene functions as an expectation maximization algorithm to determine a bounding boxes' ground truth
        probabilistically, based on the paper by Dawid and Skene 1979.
        (link to the paper: http://crowdsourcing-class.org/readings/downloads/ml/EM.pdf)

        To resume the algorithm the class inherits from the Majority Voting class to first off determine which bounding
        boxes to focus on.

        The implemented code is based on: https://github.com/dallascard/dawid_skene/blob/master/dawid_skene.py
        """
        assert sys.version_info >= (3, 7), "Python version needs to be at least 3.7 or higher. Since dawid and skene relies " \
                                            "on ordered dictonaries that have been introduced at 3.7, dawid and skene will not " \
                                            "provide reliable results."
        self.return_confusion_matrix = return_confusion_matrix
        super(LocalizationAwareEM, self).__init__(new_ann_path, annotator_key, iou_threshold, merging_ops, mask_threshold_ratio_factor)

    def process_image_annotations(self, images, annotations):
        """ Outer loop for a single image to execute the majority voting.
            Takes the images and annotations of the annotators and returns an image with a new id and the majority
            voted annotations (also with a new id).
        """
        number_of_annotators = len(images)
        min_agreement = number_of_annotators/2.0 # minimum number of annotators to agree on an element to be a mj voted gt
        new_annotations = []

        # create new image with a new unique id
        new_image = deepcopy(next(iter(images.values())))
        new_image_id = next(self.new_img_id_generator)
        new_image["id"] = new_image_id
        new_image[self.annotator_key] = "DawidSkene"
        new_image["observer_lst"] = [image[self.annotator_key] for image in images.values()]
        width = new_image["width"]
        height = new_image["height"]

        # try to find matching annotations starting with the most annotator going to the least possible number
        # of annotators
        while number_of_annotators >= min_agreement and len(annotations) > 0:
            coder_to_annotations_dict = defaultdict(list)
            for annotation in annotations:
                coder_to_annotations_dict[annotation[self.annotator_key]].append(annotation)
            # create combinations
            current_combinations = list(combinations(coder_to_annotations_dict, number_of_annotators))
            # select combination annotations - this might return a huge list
            boxes_to_fuse = []
            for combination in current_combinations:
                if len(combination) > 1:
                    current_annotation_combinations = list(product(*[coder_to_annotations_dict[annotator] for annotator in combination]))
                else:
                    current_annotation_combinations = coder_to_annotations_dict[combination[0]]
                boxes_to_fuse += current_annotation_combinations
            # retrieve newest mjv annotations and remove the ones used
            # --- this is the core logic of the majority voting ---
            fused_boxes = self.voting(boxes_to_fuse, min_agreement, height, width)
            # add annotations to the mjv annotations and exclude such selected annotations from the future mjv process
            # this uses the property that dictionaries are ordered since python 3.7
            remaining_ids = set([annotation["id"] for lst in coder_to_annotations_dict.values() for annotation in lst])
            for iou_and_ids, mjv_ann in reversed(fused_boxes.items()):
                ids = set(iou_and_ids[1:])
                if ids.issubset(remaining_ids):
                    mjv_ann["image_id"] = new_image_id
                    new_annotations.append(mjv_ann)
                    remaining_ids = remaining_ids.difference(ids)
            annotations = list(filter(lambda annotation: annotation["id"] in remaining_ids, annotations))
            # decrease the number of annotator by one
            number_of_annotators -= 1

        return new_image, new_annotations

    def post_process_annotations(self, coco_annotations):
        #the Dawid and Skene algorithm begins here
        instances, annotators, classes, counts, class_marginals, error_rates, dataset_classes = self.em_algorithm(coco_annotations)

        if self.return_confusion_matrix:
            self.annotators = annotators
            self.classes = classes
            self.error_rates = error_rates

        #
        for annotation in coco_annotations["annotations"]:
            questions = annotation["question_id"]
            likelihood_dict = defaultdict(float)
            # some the error_rates (confidences) per class up and take the one with the highest confidence
            # this should mean that for most cases the number of annotators is the deciding factor.
            # It is, however, possible that a single observer with very high confidence can outvote two very bad
            # observer.
            for annotator, question in questions.items():
                likelihood = error_rates[ annotators[annotator], classes[question], classes[question] ]
                likelihood_dict[question] += likelihood
            maximum_likeli_category = max(likelihood_dict, key=likelihood_dict.get)
            annotation["category_id"] = maximum_likeli_category

        return coco_annotations

    def voting(self, boxes_to_fuse, min_agreement, height, width):
        """ Here the voting procedure is executed
        """
        matches = {}
        for boxes in boxes_to_fuse:
            if isinstance(boxes, tuple):
                iou_bbox = self.iou_bbox(boxes)
                if iou_bbox < self.iou_threshold:
                    continue
                if self.iou_segm(boxes, height, width) < self.mask_threshold_ratio_factor:
                    continue
                # for dawid and skene we add all annotation that have been fit during the first stage
                # the exact classes will be evaluated by the question_id
                class_dict = {annotation[self.annotator_key]: annotation["category_id"] for annotation in boxes}
                ann_ids = [box["id"] for box in boxes]
                matches[tuple([iou_bbox] + ann_ids)] = self.merge_annotations(boxes, class_dict, height, width)
            else:
                # for david and skene we add all single annotations
                if min_agreement <= 1.0:
                    new_annotation = deepcopy(boxes)
                    new_ann_id = next(self.new_annotation_id_generator)
                    new_annotation["id"] = new_ann_id
                    new_annotation[self.annotator_key] = "DawidSkene"
                    new_annotation["category_id"] = None
                    new_annotation["question_id"] = {boxes[self.annotator_key]: boxes["category_id"]}
                    matches[tuple([self.iou_threshold] + [boxes["id"]])] = new_annotation
        return matches

    def merge_annotations(self, annotations, class_dict, height, width):
        votingAnnotators = {}
        annLabel = []
        annLabel2 = []

        annLabel.append(annotations[0]["category_id"])
        annLabel2.append(annotations[1]["category_id"])

        votingAnnotators[annotations[0][self.annotator_key]] = annLabel
        votingAnnotators[annotations[1][self.annotator_key]] = annLabel2

        annA = annotations[0]

        if self.merging_ops == "union":
            new_bbox = annA["bbox"]
            new_segm = self.get_segmentation_mask(annA, height, width)
        if self.merging_ops == "intersection":
            new_bbox = annA["bbox"]
            new_segm = self.get_segmentation_mask(annA, height, width)
        for annB in annotations[1:]:
            if self.merging_ops == "union":
                new_bbox = self.union_bbox(new_bbox, annB["bbox"])
                new_segm = new_segm.union(self.get_segmentation_mask(annB, height, width))
            if self.merging_ops == "intersection":
                new_bbox, _ = self.intersection_bbox_and_area(new_bbox, annB["bbox"])
                new_segm = new_segm.intersection(self.get_segmentation_mask(annB, height, width))

        if self.merging_ops == "averaging":
            bbox_conf_list = [[ann["bbox"], 1.0] for ann in annotations]
            _, new_bbox = self.get_weighted_box(bbox_conf_list, width, height)
            segm_conf_list = [[self.get_segmentation_mask(ann, height, width), 1.0] for ann in annotations]
            new_segm, _ = self.get_weighted_segmentation(segm_conf_list)

        new_annotation = deepcopy(annA)
        new_ann_id = next(self.new_annotation_id_generator)
        new_annotation["id"] = new_ann_id
        new_annotation["bbox"] = new_bbox
        if self.merging_ops == "averaging":
            new_annotation["segmentation"] = new_segm
        else:
            new_annotation["segmentation"] = self.get_coco_segmentation(new_segm)
        new_annotation["category_id"] = None
        new_annotation[self.annotator_key] = "DawidSkene"
        new_annotation["question_id"] = class_dict

        return new_annotation

    def em_algorithm(self, responses, tol=0.001, max_iter=100, init='average'):
        """
            Run the Dawid-Skene estimator on response data
        Input:
            questions: a dictionary object of responses:
                {Questions: {Annotators: [Classlabel]}}
            tol: tolerance required for convergence of EM
            max_iter: maximum number of iterations of EM
        """
        # convert responses to counts
        (instances, annotators, classes, counts) = self.instances_to_counts(responses)

        # initialize
        iter = 0
        converged = False
        old_class_marginals = None
        old_error_rates = None
        dataset_classes = self.initialize(counts)

        # while not converged do:
        while not converged:
            iter += 1
            # M-step
            (class_marginals, error_rates) = self.m_step(counts, dataset_classes)
            # E-setp
            dataset_classes = self.e_step(counts, class_marginals, error_rates)

            # check likelihood
            log_L = self.calc_likelihood(counts, class_marginals, error_rates)

            # check for convergence
            if old_class_marginals is not None:
                class_marginals_diff = np.sum(np.abs(class_marginals - old_class_marginals))
                error_rates_diff = np.sum(np.abs(error_rates - old_error_rates))
                print(iter, '\t', log_L, '\t%.6f\t%.6f' % (class_marginals_diff, error_rates_diff))
                if (class_marginals_diff < tol and error_rates_diff < tol) or iter > max_iter:
                    converged = True
            else:
                print(iter, '\t', log_L)

            # update current values
            old_class_marginals = class_marginals
            old_error_rates = error_rates

        return (instances, annotators, classes, counts, class_marginals, error_rates, dataset_classes)

    def instances_to_counts(self, coco_annotations):
        """
        Function: instances_to_counts()
            Convert a matrix of annotations to count data
        Return:
            instances: list of annotations / instances
            observers: list of annotators
            classes: list of annotated classes
            counts: 3d array of counts: [instances x annotators x classes]
        """

        instances = [annotation["id"] for annotation in coco_annotations["annotations"]]
        # patients.sort()
        instances = sorted(instances)
        nInstances = len(instances)
        # determine the observers and classes

        classes = sorted( [category["id"] for category in coco_annotations["categories"]] )
        classes = {cls: id for id, cls in enumerate(classes)}
        nClasses = len(classes)

        responses = [annotation["question_id"] for annotation in coco_annotations["annotations"]]
        observers = sorted( list( { coder for coders in responses for coder in coders.keys() } ) )
        observers = {observer: id for id, observer in enumerate(observers)}
        nObservers = len(observers)

        # create a 3d array to hold counts
        counts = np.zeros([nInstances, nObservers, nClasses])

        # convert responses to counts
        for position, response in enumerate(responses):
            for observer, cls in response.items():
                # three dimensions for the 3d array
                # 1st: instance = position
                # 2nd: annotator or observer
                # 3th: class number
                # increment by one for every occurrence for the current page
                counts[position, observers[observer], classes[cls]] += 1

        return (instances, observers, classes, counts)

    def initialize(self, counts):
        """
            Get initial estimates for the true classes
        Input:
            counts: counts of the number of times each vote was received
                by each annotator from each question: [questions x annotators x classes]
        Returns:
            question_classes: matrix of estimates of true classes:
                [question x annotations]
        """
        [nQuestion, nAnnotator, nClasses] = np.shape(counts)
        # sum over observers
        response_sums = np.sum(counts, 1)
        # create an empty array
        question_classes = np.zeros([nQuestion, nClasses])
        # for each patient, take the average number of observations in each class
        for p in range(nQuestion):
            question_classes[p, :] = response_sums[p, :] / np.sum(response_sums[p, :], dtype=float)

        return question_classes

    def m_step(self, counts, patient_classes):
        """
            Get estimates for the prior class probabilities (p_j) and the error
            rates (pi_jkl) using MLE with current estimates of true classes
        Input:
            counts: Array of how many times each vote was received
                by each annotator from each question
            dataset_classes: Matrix of current assignments of votes to classes
        Returns:
            p_j: class marginals [classes]
            pi_kjl: error rates - the probability of annotator k receiving
                vote l from a question in class j [observers, classes, classes]
        """
        [nQuestions, nAnnotators, nClasses] = np.shape(counts)

        # compute class marginals
        class_marginals = np.sum(patient_classes, 0) / float(nQuestions)

        # compute error rates
        error_rates = np.zeros([nAnnotators, nClasses, nClasses])
        for k in range(nAnnotators):
            for j in range(nClasses):
                for l in range(nClasses):
                    error_rates[k, j, l] = np.dot(patient_classes[:, j], counts[:, k, l])
                # normalize by summing over all observation classes
                sum_over_responses = np.sum(error_rates[k, j, :])
                if sum_over_responses > 0:
                    error_rates[k, j, :] = error_rates[k, j, :] / float(sum_over_responses)

        return (class_marginals, error_rates)

    def e_step(self, counts, class_marginals, error_rates):
        """
            Determine the probability of each question belonging to each class,
            given current ML estimates of the parameters from the M-step
        Inputs:
            counts: Array of how many times each votes was received
                by each annotator from each question
            class_marginals: probability of a random question belonging to each class
            error_rates: probability of annotator k assigning a question in class j
                to class l [annotator, classes, classes]
        Returns:
            dataset_classes: Soft assignments of questions to classes
                [questions x classes]
        """
        [nQuestions, nAnnotators, nClasses] = np.shape(counts)

        dataset_classes = np.zeros([nQuestions, nClasses])

        for i in range(nQuestions):
            for j in range(nClasses):
                estimate = class_marginals[j]
                estimate *= np.prod(np.power(error_rates[:, j, :], counts[i, :, :]))

                dataset_classes[i, j] = estimate
            # normalize error rates by dividing by the sum over all observation classes
            vote_sum = np.sum(dataset_classes[i, :])
            if vote_sum > 0:
                dataset_classes[i, :] = dataset_classes[i, :] / float(vote_sum)

        return dataset_classes

    def calc_likelihood(self, counts, class_marginals, error_rates):
        """
            Calculate the likelihood given the current parameter estimates
        Inputs:
            counts: Array of how many times each vote was received
                by each annotator from each question
            class_marginals: probability of a random question belonging to each class
            error_rates: probability of annotator k assigning a question in class j
                to class l [annotator, classes, classes]
        Returns:
            Likelihood given current parameter estimates
        """
        [nQuestions, nAnnotators, nClasses] = np.shape(counts)
        log_L = 0.0

        for i in range(nQuestions):
            question_likelihood = 0.0
            for j in range(nClasses):
                class_prior = class_marginals[j]
                question_class_likelihood = np.prod(np.power(error_rates[:, j, :], counts[i, :, :]))
                question_class_posterior = class_prior * question_class_likelihood
                question_likelihood += question_class_posterior
            temp = log_L + np.log(question_likelihood)

            if np.isnan(temp) or np.isinf(temp):
                sys.exit()
            log_L = temp

        return log_L
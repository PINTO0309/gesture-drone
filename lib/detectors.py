import cv2
import os
import sys
from logging import getLogger, basicConfig, DEBUG, INFO
from openvino.inference_engine import IENetwork, IEPlugin
from timeit import default_timer as timer
import numpy as np
import queue

logger = getLogger(__name__)
basicConfig(
    level=INFO,
    format="%(asctime)s %(levelname)s %(name)s %(funcName)s(): %(message)s")

is_myriad_plugin_initialized = False
myriad_plugin = None
is_cpu_plugin_initialized = False
cpu_plugin = None

CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
    "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))


class BaseDetection(object):
    def __init__(self, device, model_xml, cpu_extension, plugin_dir,
                 detection_of):
        # Each device's plugin should be initialized only once,
        # MYRIAD plugin would be failed when createting exec_net(plugin.load method)
        # Error: "RuntimeError: Can not init USB device: NC_DEVICE_NOT_FOUND"
        global is_myriad_plugin_initialized
        global myriad_plugin
        global is_cpu_plugin_initialized
        global cpu_plugin

        if device == 'MYRIAD' and not is_myriad_plugin_initialized:
            self.plugin = self._init_plugin(device, cpu_extension, plugin_dir)
            is_myriad_plugin_initialized = True
            myriad_plugin = self.plugin
        elif device == 'MYRIAD' and is_myriad_plugin_initialized:
            self.plugin = myriad_plugin
        elif device == 'CPU' and not is_cpu_plugin_initialized:
            self.plugin = self._init_plugin(device, cpu_extension, plugin_dir)
            is_cpu_plugin_initialized = True
            cpu_plugin = self.plugin
        elif device == 'CPU' and is_cpu_plugin_initialized:
            self.plugin = cpu_plugin
        else:
            self.plugin = self._init_plugin(device, cpu_extension, plugin_dir)

        # Read IR
        self.net = self._read_ir(model_xml, detection_of)
        # Load IR model to the plugin
        self.input_blob, self.out_blob, self.exec_net, self.input_dims, self.output_dims = self._load_ir_to_plugin(
            device, detection_of)

    def _init_plugin(self, device, cpu_extension, plugin_dir):
        logger.info("Initializing plugin for {} device...".format(device))
        plugin = IEPlugin(device=device, plugin_dirs=plugin_dir)
        logger.info(
            "Plugin for {} device version:{}".format(device, plugin.version))
        if cpu_extension and 'CPU' in device:
            plugin.add_cpu_extension(cpu_extension)
        return plugin

    def _read_ir(self, model_xml, detection_of):
        logger.info("Reading IR Loading for {}...".format(detection_of))
        model_bin = os.path.splitext(model_xml)[0] + ".bin"
        return IENetwork(model=model_xml, weights=model_bin)

    def _load_ir_to_plugin(self, device, detection_of):
        if device == "CPU" and detection_of == "Face Detection":
            supported_layers = self.plugin.get_supported_layers(self.net)
            not_supported_layers = [
                l for l in self.net.layers.keys() if l not in supported_layers
            ]
            if len(not_supported_layers) != 0:
                logger.error(
                    "Following layers are not supported by the plugin for specified device {}:\n {}".
                    format(self.plugin.device, ', '.join(
                        not_supported_layers)))
                logger.error(
                    "Please try to specify cpu extensions library path in demo's command line parameters using -l "
                    "or --cpu_extension command line argument")
                sys.exit(1)
        if detection_of == "Face Detection":
            logger.info("Checking Face Detection network inputs")
            assert len(self.net.inputs.keys(
            )) == 1, "Face Detection network should have only one input"
            logger.info("Checking Face Detection network outputs")
            assert len(
                self.net.outputs
            ) == 1, "Face Detection network should have only one output"

        input_blob = next(iter(self.net.inputs))
        out_blob = next(iter(self.net.outputs))
        logger.info("Loading {} model to the {} plugin...".format(
            device, detection_of))
        exec_net = self.plugin.load(network=self.net, num_requests=2)
        input_dims = self.net.inputs[input_blob].shape
        output_dims = self.net.outputs[out_blob].shape
        logger.info("{} input dims:{} output dims:{} ".format(
            detection_of, input_dims, output_dims))
        return input_blob, out_blob, exec_net, input_dims, output_dims

    def submit_req(self, face_frame, next_face_frame, is_async_mode):
        n, c, h, w = self.input_dims

        if is_async_mode:
            logger.debug(
                "*** start_async *** cur_req_id:{} next_req_id:{} async:{}".
                format(self.cur_request_id, self.next_request_id,
                       is_async_mode))
            in_frame = cv2.resize(next_face_frame, (w, h))
            # Change data layout from HWC to CHW
            in_frame = in_frame.transpose((2, 0, 1))
            in_frame = in_frame.reshape((n, c, h, w))
            self.exec_net.start_async(
                request_id=self.next_request_id,
                inputs={self.input_blob: in_frame})
        else:
            logger.debug(
                "*** start_sync *** cur_req_id:{} next_req_id:{} async:{}".
                format(self.cur_request_id, self.next_request_id,
                       is_async_mode))
            self.exec_net.requests[self.cur_request_id].wait(-1)
            in_frame = cv2.resize(face_frame, (w, h))
            # Change data layout from HWC to CHW
            in_frame = in_frame.transpose((2, 0, 1))
            in_frame = in_frame.reshape((n, c, h, w))
            self.exec_net.start_async(
                request_id=self.cur_request_id,
                inputs={self.input_blob: in_frame})

    def wait(self):
        logger.debug("*** start wait:{} ***".format(self.exec_net.requests[
            self.cur_request_id].wait(-1)))

        if self.exec_net.requests[self.cur_request_id].wait(-1) == 0:
            return True
        else:
            return False


class FaceDetection(BaseDetection):
    def __init__(self, device, model_xml, cpu_extension, plugin_dir,
                 prob_threshold_face, is_async_mode):
        self.prob_threshold_face = prob_threshold_face
        detection_of = "Face Detection"
        super().__init__(device, model_xml, cpu_extension, plugin_dir,
                         detection_of)

        del self.net
        self.cur_request_id = 0
        self.next_request_id = 1

    def get_results(self, is_async_mode):
        """
        The net outputs a blob with shape: [1, 1, 200, 7]
        The description has the format: [image_id, label, conf, x_min, y_min, x_max, y_max]
        """

        faces = None

        res = self.exec_net.requests[self.cur_request_id].outputs[
            self.out_blob]  # res's shape: [1, 1, 200, 7]

        # Get rows whose confidence is larger than prob_threshold.
        # detected faces are also used by age/gender, emotion, landmark, head pose detection.
        faces = res[0][:, np.where(res[0][0][:, 2] > self.prob_threshold_face)]

        if is_async_mode:
            self.cur_request_id, self.next_request_id = self.next_request_id, self.cur_request_id

        return faces


class AgeGenderDetection(BaseDetection):
    def __init__(self, device, model_xml, cpu_extension, plugin_dir,
                 prob_threshold, is_async_mode):
        detection_of = "Age/Gender Detection"
        super().__init__(device, model_xml, cpu_extension, plugin_dir,
                         detection_of)
        self.label = ('F', 'M')
        del self.net
        self.cur_request_id = 0
        self.next_request_id = 1

    def get_results(self, is_async_mode):
        """
        Output layer names in Inference Engine format:
         "age_conv3", shape: [1, 1, 1, 1] - Estimated age divided by 100.
         "prob", shape: [1, 2, 1, 1] - Softmax output across 2 type classes [female, male]
        """
        age = 0
        gender = ""
        logger.debug("*** get_results start *** cur_request_id:{}".format(
            self.cur_request_id))
        age = self.exec_net.requests[self.cur_request_id].outputs['age_conv3']
        prob = self.exec_net.requests[self.cur_request_id].outputs['prob']
        age = age[0][0][0][0] * 100
        gender = self.label[np.argmax(prob[0])]
        if is_async_mode:
            self.cur_request_id, self.next_request_id = self.next_request_id, self.cur_request_id
        return age, gender


class EmotionsDetection(BaseDetection):
    def __init__(self, device, model_xml, cpu_extension, plugin_dir,
                 prob_threshold, is_async_mode):
        detection_of = "Emotion Detection"
        super().__init__(device, model_xml, cpu_extension, plugin_dir,
                         detection_of)
        self.label = ('neutral', 'happy', 'sad', 'surprise', 'anger')
        del self.net
        self.cur_request_id = 0
        self.next_request_id = 1

    def get_results(self, is_async_mode):
        """
        Output layer names in Inference Engine format:
         "prob_emotion", shape: [1, 5, 1, 1]
         - Softmax output across five emotions ('neutral', 'happy', 'sad', 'surprise', 'anger').
        """

        emotion = ""
        res = self.exec_net.requests[self.cur_request_id].outputs[
            self.out_blob]
        emotion = self.label[np.argmax(res[0])]

        if is_async_mode:
            self.cur_request_id, self.next_request_id = self.next_request_id, self.cur_request_id

        return emotion


class HeadPoseDetection(BaseDetection):
    def __init__(self, device, model_xml, cpu_extension, plugin_dir,
                 prob_threshold, is_async_mode):
        detection_of = "Head Pose Detection"
        super().__init__(device, model_xml, cpu_extension, plugin_dir,
                         detection_of)
        del self.net
        self.cur_request_id = 0
        self.next_request_id = 1

    def get_results(self, is_async_mode):
        """
        Output layer names in Inference Engine format:
         "angle_y_fc", shape: [1, 1] - Estimated yaw (in degrees).
         "angle_p_fc", shape: [1, 1] - Estimated pitch (in degrees).
         "angle_r_fc", shape: [1, 1] - Estimated roll (in degrees).
        Each output contains one float value that represents value in 
        Tait-Bryan angles (yaw, pitсh or roll).
        """

        yaw = .0  # Axis of rotation: z
        pitch = .0  # Axis of rotation: y
        roll = .0  # Axis of rotation: x

        yaw = self.exec_net.requests[self.cur_request_id].outputs[
            'angle_y_fc'][0][0]
        pitch = self.exec_net.requests[self.cur_request_id].outputs[
            'angle_p_fc'][0][0]
        roll = self.exec_net.requests[self.cur_request_id].outputs[
            'angle_r_fc'][0][0]

        if is_async_mode:
            self.cur_request_id, self.next_request_id = self.next_request_id, self.cur_request_id

        return yaw, pitch, roll


class FacialLandmarksDetection(BaseDetection):
    def __init__(self, device, model_xml, cpu_extension, plugin_dir,
                 prob_threshold, is_async_mode):
        detection_of = "Facial Landmarks Detection"
        super().__init__(device, model_xml, cpu_extension, plugin_dir,
                         detection_of)
        del self.net
        self.cur_request_id = 0
        self.next_request_id = 1

    def get_results(self, is_async_mode):
        """
        # Output layer names in Inference Engine format:
        # landmarks-regression-retail-0009:
        #   "95", [1, 10, 1, 1], containing a row-vector of 10 floating point values for five landmarks
        #         coordinates in the form (x0, y0, x1, y1, ..., x5, y5).
        #         All the coordinates are normalized to be in range [0,1]
        # facial-landmarks-35-adas-0001:
        #   "align_fc3", [1,70], the shape: [1, 70], containing row-vector of 70 floating point values for 35 landmarks'
        #    normed coordinates in the form (x0, y0, x1, y1, ..., x34, y34).
        """

        normed_landmarks = np.zeros(0)

        if self.output_dims == [1, 10, 1, 1]:
            # for landmarks-regression_retail-0009
            normed_landmarks = self.exec_net.requests[
                self.cur_request_id].outputs[self.out_blob].reshape(1, 10)[0]
        else:
            # for facial-landmarks-35-adas-0001
            normed_landmarks = self.exec_net.requests[
                self.cur_request_id].outputs[self.out_blob][0]

        if is_async_mode:
            self.cur_request_id, self.next_request_id = self.next_request_id, self.cur_request_id

        return normed_landmarks


class SSDetection(BaseDetection):
    def __init__(self, device, model_xml, cpu_extension, plugin_dir,
                 prob_threshold, is_async_mode):
        detection_of = "MobileNet-SSD Detection"
        super().__init__(device, model_xml, cpu_extension, plugin_dir,
                         detection_of)
        del self.net
        self.prob_threshold = prob_threshold
        self.cur_request_id = 0
        self.next_request_id = 1

    def object_inference(self, frame, next_frame, is_async_mode):
        n, c, h, w = self.input_dims
        frame_h, frame_w = frame.shape[:2]  # shape (h, w, c)
        det_time = 0

        # Main sync point:
        # in the truly Async mode we start the NEXT infer request, while waiting for the CURRENT to complete
        # in the regular mode we start the CURRENT request and immediately wait for it's completion
        inf_start = timer()
        if is_async_mode:
            in_frame = cv2.dnn.blobFromImage(
                cv2.resize(next_frame, (300, 300)), 0.007843, (300, 300),
                127.5)
            self.exec_net.start_async(
                request_id=self.next_request_id,
                inputs={self.input_blob: in_frame})
        else:
            if self.exec_net.requests[self.cur_request_id].wait(-1) == 0:
                in_frame = cv2.dnn.blobFromImage(
                    cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
                self.exec_net.start_async(
                    request_id=self.cur_request_id,
                    inputs={self.input_blob: in_frame})
        if self.exec_net.requests[self.cur_request_id].wait(-1) == 0:
            inf_end = timer()
            det_time = inf_end - inf_start
            # Parse detection results of the current request
            logger.debug("computing object detections...")
            det_objects = self.exec_net.requests[self.cur_request_id].outputs[
                self.out_blob]
            # loop over the detections
            for i in np.arange(0, det_objects.shape[2]):
                # extract the confidence (i.e., probability) associated with the prediction
                confidence = det_objects[0, 0, i, 2]
                # filter out weak detections by ensuring the `confidence` is greater than the minimum confidence
                if confidence > self.prob_threshold:
                    # extract the index of the class label from the `detections`, then compute the (x, y)-coordinates
                    # of the bounding box for the object
                    idx = int(det_objects[0, 0, i, 1])
                    box = det_objects[0, 0, i, 3:7] * np.array(
                        [frame_w, frame_h, frame_w, frame_h])
                    (startX, startY, endX, endY) = box.astype("int")
                    logger.debug("startX, startY, endX, endY: {}".format(
                        box.astype("int")))

                    # display the prediction
                    label = "{}: {:.2f}%".format(CLASSES[idx],
                                                 confidence * 100)
                    logger.debug("{} {}".format(self.cur_request_id, label))
                    cv2.rectangle(frame, (startX, startY), (endX, endY),
                                  COLORS[idx], 2)
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    cv2.putText(frame, label, (startX, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

        if is_async_mode:
            self.cur_request_id, self.next_request_id = self.next_request_id, self.cur_request_id

        return det_time, frame

class PoseDetection(BaseDetection):
    def __init__(self, device, model_xml, cpu_extension, plugin_dir,
                 prob_threshold, is_async_mode):
        self.prob_threshold = prob_threshold
        detection_of = "Pose Estimation"
        super().__init__(device, model_xml, cpu_extension, plugin_dir,
                         detection_of)

        del self.net
        self.cur_request_id = 0
        self.next_request_id = 1
        self.w = 432
        self.h = 368
        self.threshold = 0.1
        self.nPoints = 18
        self.keypointsMapping = ['Nose', 'Neck', 'R-Sho', 'R-Elb', 'R-Wr', 'L-Sho', 'L-Elb', 'L-Wr', 'R-Hip', 'R-Knee', 'R-Ank', 'L-Hip', 'L-Knee', 'L-Ank', 'R-Eye', 'L-Eye', 'R-Ear', 'L-Ear']
        self.POSE_PAIRS = [[1,2], [1,5], [2,3], [3,4], [5,6], [6,7], [1,8], [8,9], [9,10], [1,11], [11,12], [12,13], [1,0], [0,14], [14,16], [0,15], [15,17], [2,17], [5,16]]
        self.mapIdx = [[31,32], [39,40], [33,34], [35,36], [41,42], [43,44], [19,20], [21,22], [23,24], [25,26], [27,28], [29,30], [47,48], [49,50], [53,54], [51,52], [55,56], [37,38], [45,46]]
        self.colors = [[0,100,255], [0,100,255], [0,255,255], [0,100,255], [0,255,255], [0,100,255], [0,255,0], [255,200,100], [255,0,255], [0,255,0], [255,200,100], [255,0,255], [0,0,255], [255,0,0], [200,200,0], [255,0,0], [200,200,0], [0,0,0]]
        self.gesture_point = queue.Queue()

    def getKeypoints(self, probMap, threshold=0.1):

        mapSmooth = cv2.GaussianBlur(probMap, (3, 3), 0, 0)
        mapMask = np.uint8(mapSmooth>threshold)
        keypoints = []
        contours = None
        #OpenCV4.x
        contours, _ = cv2.findContours(mapMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            blobMask = np.zeros(mapMask.shape)
            blobMask = cv2.fillConvexPoly(blobMask, cnt, 1)
            maskedProbMap = mapSmooth * blobMask
            _, maxVal, _, maxLoc = cv2.minMaxLoc(maskedProbMap)
            keypoints.append(maxLoc + (probMap[maxLoc[1], maxLoc[0]],))

        return keypoints


    def getValidPairs(self, outputs, w, h, detected_keypoints):
        valid_pairs = []
        invalid_pairs = []
        n_interp_samples = 10
        paf_score_th = 0.1
        conf_th = 0.7

        for k in range(len(self.mapIdx)):
            pafA = outputs[0, self.mapIdx[k][0], :, :]
            pafB = outputs[0, self.mapIdx[k][1], :, :]
            pafA = cv2.resize(pafA, (w, h))
            pafB = cv2.resize(pafB, (w, h))

            candA = detected_keypoints[self.POSE_PAIRS[k][0]]
            candB = detected_keypoints[self.POSE_PAIRS[k][1]]
            nA = len(candA)
            nB = len(candB)

            if( nA != 0 and nB != 0):
                valid_pair = np.zeros((0,3))
                for i in range(nA):
                    max_j=-1
                    maxScore = -1
                    found = 0
                    for j in range(nB):
                        d_ij = np.subtract(candB[j][:2], candA[i][:2])
                        norm = np.linalg.norm(d_ij)
                        if norm:
                            d_ij = d_ij / norm
                        else:
                            continue
                        interp_coord = list(zip(np.linspace(candA[i][0], candB[j][0], num=n_interp_samples),
                                                np.linspace(candA[i][1], candB[j][1], num=n_interp_samples)))
                        paf_interp = []
                        for k in range(len(interp_coord)):
                            paf_interp.append([pafA[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))],
                                               pafB[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))] ])
                        paf_scores = np.dot(paf_interp, d_ij)
                        avg_paf_score = sum(paf_scores)/len(paf_scores)

                        if ( len(np.where(paf_scores > paf_score_th)[0]) / n_interp_samples ) > conf_th :
                            if avg_paf_score > maxScore:
                                max_j = j
                                maxScore = avg_paf_score
                                found = 1
                    if found:
                        valid_pair = np.append(valid_pair, [[candA[i][3], candB[max_j][3], maxScore]], axis=0)

                valid_pairs.append(valid_pair)
            else:
                invalid_pairs.append(k)
                valid_pairs.append([])
        return valid_pairs, invalid_pairs


    def getPersonwiseKeypoints(self, valid_pairs, invalid_pairs, keypoints_list):
        personwiseKeypoints = -1 * np.ones((0, 19))

        for k in range(len(self.mapIdx)):
            if k not in invalid_pairs:
                partAs = valid_pairs[k][:,0]
                partBs = valid_pairs[k][:,1]
                indexA, indexB = np.array(self.POSE_PAIRS[k])

                for i in range(len(valid_pairs[k])):
                    found = 0
                    person_idx = -1
                    for j in range(len(personwiseKeypoints)):
                        if personwiseKeypoints[j][indexA] == partAs[i]:
                            person_idx = j
                            found = 1
                            break

                    if found:
                        personwiseKeypoints[person_idx][indexB] = partBs[i]
                        personwiseKeypoints[person_idx][-1] += keypoints_list[partBs[i].astype(int), 2] + valid_pairs[k][i][2]

                    elif not found and k < 17:
                        row = -1 * np.ones(19)
                        row[indexA] = partAs[i]
                        row[indexB] = partBs[i]
                        row[-1] = sum(keypoints_list[valid_pairs[k][i,:2].astype(int), 2]) + valid_pairs[k][i][2]
                        personwiseKeypoints = np.vstack([personwiseKeypoints, row])
        return personwiseKeypoints


    def object_inference(self, frame, next_frame, is_async_mode):
        det_time = 0
        command = ""

        colw = frame.shape[1] #480
        colh = frame.shape[0] #360
        new_w = int(colw * min(self.w/colw, self.h/colh))
        new_h = int(colh * min(self.w/colw, self.h/colh))

        resized_image = cv2.resize(frame, (new_w, new_h), interpolation = cv2.INTER_CUBIC)
        canvas = np.full((self.h, self.w, 3), 128)
        canvas[(self.h - new_h)//2:(self.h - new_h)//2 + new_h,(self.w - new_w)//2:(self.w - new_w)//2 + new_w, :] = resized_image

        prepimg = canvas
        prepimg = prepimg[np.newaxis, :, :, :]     # Batch size axis add
        prepimg = prepimg.transpose((0, 3, 1, 2))  # NHWC to NCHW, (1, 3, 368, 432)

        if is_async_mode:
            self.exec_net.start_async(request_id=self.next_request_id, inputs={self.input_blob: prepimg})
        else:
            if self.exec_net.requests[self.cur_request_id].wait(-1) == 0:
                self.exec_net.start_async(request_id=self.cur_request_id, inputs={self.input_blob: prepimg})

        if self.exec_net.requests[self.cur_request_id].wait(-1) == 0:
            inf_start = timer()
            #outputs = self.exec_net.infer(inputs={self.input_blob: prepimg})["Openpose/concat_stage7"]
            outputs = self.exec_net.requests[self.cur_request_id].outputs["Openpose/concat_stage7"]
            inf_end = timer()
            det_time = inf_end - inf_start

            detected_keypoints = []
            keypoints_list = np.zeros((0, 3))
            keypoint_id = 0

            for part in range(self.nPoints):
                probMap = outputs[0, part, :, :]
                probMap = cv2.resize(probMap, (canvas.shape[1], canvas.shape[0])) # (432, 368)
                keypoints = self.getKeypoints(probMap, self.threshold)
                #print("Keypoints - {} - {} : {}".format(self.keypointsMapping[part], part, keypoints))
                keypoints_with_id = []

                for i in range(len(keypoints)):
                    keypoints_with_id.append(keypoints[i] + (keypoint_id, part))
                    keypoints_list = np.vstack([keypoints_list, keypoints[i]])
                    keypoint_id += 1

                detected_keypoints.append(keypoints_with_id)

            frameClone = np.uint8(canvas.copy())
            for i in range(self.nPoints):
                for j in range(len(detected_keypoints[i])):
                    #detected_keypoints[i][0] = (x, y, prob, keypoint_id, part)

                    keypointsXY = detected_keypoints[i][j][0:2]
                    if detected_keypoints[i][j][4] == 4:
                        X = keypointsXY[1]
                        Y = keypointsXY[0]
                        self.gesture_point.put((X, Y))
                        qsize = self.gesture_point.qsize()

                        if qsize > 10:

                            point_list = []

                            for q in range(qsize):
                                X, Y = self.gesture_point.get()

                                # A area start judgment
                                if X >= 0 and X <= (new_w // 2)-1:
                                    if len(point_list) > 0:
                                        if not point_list[len(point_list)-1] == "A":
                                            point_list.append("A")
                                    else:
                                        point_list.append("A")

                                if X >= (new_w // 2) and X <= new_w:
                                    if len(point_list) > 0:
                                        if not point_list[len(point_list)-1] == "B":
                                            point_list.append("B")
                                    else:
                                        point_list.append("B")

                            # Flip direction judgment
                            self.gesture_point.queue.clear()
                            print(point_list)
                            if len(point_list) >= 3:
                                print(''.join(point_list[0:3]))
                                if   ''.join(point_list[0:3]) == "ABA":
                                    command = "rflip"
                                elif ''.join(point_list[0:3]) == "BAB":
                                    command = "lflip"
                            else:
                                command = "none"

                        cv2.circle(frameClone, keypointsXY, 5, [255,255,255], -1, cv2.LINE_AA)

                    else:
                        cv2.circle(frameClone, keypointsXY, 5, self.colors[i], -1, cv2.LINE_AA)

            valid_pairs, invalid_pairs = self.getValidPairs(outputs, self.w, self.h, detected_keypoints)
            personwiseKeypoints = self.getPersonwiseKeypoints(valid_pairs, invalid_pairs, keypoints_list)

            for i in range(17):
                for n in range(len(personwiseKeypoints)):
                    index = personwiseKeypoints[n][np.array(self.POSE_PAIRS[i])]
                    if -1 in index:
                        continue
                    B = np.int32(keypoints_list[index.astype(int), 0])
                    A = np.int32(keypoints_list[index.astype(int), 1])
                    cv2.line(frameClone, (B[0], A[0]), (B[1], A[1]), self.colors[i], 3, cv2.LINE_AA)

        if is_async_mode:
            self.cur_request_id, self.next_request_id = self.next_request_id, self.cur_request_id

        return det_time, frameClone, command



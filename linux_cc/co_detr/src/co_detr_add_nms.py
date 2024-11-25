
'''

xujing

2024-10-2

mmdetection=3.3.0

co-detr onnx添加nms plugin

'''

import onnx_graphsurgeon as gs
import numpy as np
import onnx


score_node_name = "Sigmoid_8437"
index_node_name = "Div_8449"
box_node_name = "ScatterND_8577"

num_class = 80


def gather_concat_score(graph,num_class=num_class):
    score_output_node = [node for node in graph.nodes if node.name == score_node_name][0]
    score_output = score_output_node.outputs[0]

    index_output_node = [node for node in graph.nodes if node.name == index_node_name][0]
    index_output = index_output_node.outputs[0]

    # gather
    gather_output = gs.Variable(name="score_input_0",shape=(300,num_class),dtype=np.float32)
    gather_node = gs.Node(op="Gather",inputs=[score_output,index_output],outputs=[gather_output])

    # # Unsqueeze
    # unsqueeze_output = gs.Variable(name="score_input",shape=(1,300,num_class),dtype=np.float32)
    # unsqueeze_node = gs.Node(op="Unsqueeze",inputs=[gather_output],outputs=[unsqueeze_output],attrs={"axes":0})
    # reshape
    shape_score = gs.Constant("shape_score",values=np.array([1,300,num_class],dtype=np.int64))
    scores = gs.Variable(name="score_input",shape=(1,300,num_class),dtype=np.float32)
    scores_node = gs.Node(op="Reshape",inputs=[gather_output,shape_score],outputs=[scores])


    # concat
    box_node =  [node for node in graph.nodes if node.name == box_node_name][0]
    # box_output = box_node.outputs[0]

    # Unsqueeze
    # unsqueeze_output_1 = gs.Variable(name="box_input",shape=(1,300,4),dtype=np.float32)
    # unsqueeze_node_1 = gs.Node(op="Unsqueeze",inputs=[box_node.outputs[0]],outputs=[unsqueeze_output_1],attrs={"axes":0})

    # 替换为reshape,不用unsqueeze
    shape_box = gs.Constant("shape_box",values=np.array([1,300,4],dtype=np.int64))  #batchnms_trt: [1,300,1,4]
    boxes = gs.Variable(name="box_input",shape=(1,300,4),dtype=np.float32)
    boxes_node = gs.Node(op="Reshape",inputs=[box_node.outputs[0],shape_box],outputs=[boxes])

    # concat_output = gs.Variable(name="concat_box",shape=(300,num_class+4),dtype=np.float32)
    # concat_node = gs.Node(op="Concat",inputs=[box_output,gather_output],outputs=[concat_output],attrs={"axis":1})

    # graph.nodes.extend([gather_node,concat_node,])
    graph.nodes.extend([gather_node,scores_node,boxes_node])

    graph.outputs = [ boxes, scores ]

    graph.cleanup().toposort()
    # onnx.save(gs.export_onnx(graph),"./last_1.onnx")

    return graph



# graph中插入EfficientNMS plugin op
def create_and_add_plugin_node(graph, max_output_boxes, nms_type="efficientnms"):

    batch_size = graph.inputs[0].shape[0]
    print("The batch size is: ", batch_size)
    # input_h = graph.inputs[0].shape[2]
    # input_w = graph.inputs[0].shape[3]

    tensors = graph.tensors()
    boxes_tensor = tensors["box_input"]
    confs_tensor = tensors["score_input"]

    print(boxes_tensor)
    print(confs_tensor)

    if nms_type == "batchnms":
        num_detections = gs.Variable(name="num_detections").to_variable(dtype=np.int32, shape=[batch_size])
        nmsed_boxes = gs.Variable(name="nmsed_boxes").to_variable(dtype=np.float32, shape=[batch_size, max_output_boxes, 4])
        nmsed_scores = gs.Variable(name="nmsed_scores").to_variable(dtype=np.float32, shape=[batch_size, max_output_boxes])
        nmsed_classes = gs.Variable(name="nmsed_classes").to_variable(dtype=np.float32, shape=[batch_size, max_output_boxes])


    elif nms_type == "efficientnms":
        num_detections = gs.Variable(name="num_detections").to_variable(dtype=np.int32, shape=[batch_size, 1])
        nmsed_boxes = gs.Variable(name="detection_boxes").to_variable(dtype=np.float32, shape=[batch_size, max_output_boxes, 4])
        nmsed_scores = gs.Variable(name="detection_scores").to_variable(dtype=np.float32, shape=[batch_size, max_output_boxes])
        nmsed_classes = gs.Variable(name="detection_classes").to_variable(dtype=np.int32, shape=[batch_size, max_output_boxes])

    new_outputs = [num_detections, nmsed_boxes, nmsed_scores, nmsed_classes]

    if nms_type == "batchnms":
        mns_node = gs.Node(
            op="BatchedNMS_TRT",
            attrs=create_attrs_batchnms(max_output_boxes),
            inputs=[boxes_tensor, confs_tensor],
            outputs=new_outputs)
    elif nms_type == "efficientnms":
        mns_node = gs.Node(
            op="EfficientNMS_TRT",
            attrs=create_attrs_efficientnms(max_output_boxes),
            inputs=[boxes_tensor, confs_tensor],
            outputs=new_outputs)

    graph.nodes.append(mns_node)
    graph.outputs = new_outputs

    return graph.cleanup().toposort()


def create_attrs_efficientnms(max_output_boxes=100):

    attrs = {}

    attrs["score_threshold"] = 0.70
    attrs["iou_threshold"] = 0.45
    attrs["max_output_boxes"] = max_output_boxes
    attrs["background_class"] = -1
    attrs["score_activation"] = False
    attrs["class_agnostic"] = False
    attrs["box_coding"] = 0
    # 001 is the default plugin version the parser will search for, and therefore can be omitted,
    # but we include it here for illustrative purposes.
    attrs["plugin_version"] = "1"

    return attrs

def create_attrs_batchnms(max_output_boxes=100):

    attrs = {}

    attrs["shareLocation"] = True
    attrs["backgroundLabelId"] = -1
    attrs["numClasses"] = 80
    attrs["topK"] = 1000
    attrs["keepTopK"] = max_output_boxes
    attrs["scoreThreshold"] = 0.25
    attrs["iouThreshold"] = 0.45

    attrs["isNormalized"] = False
    attrs["clipBoxes"] = False
    attrs["scoreBits"] = 16  #FP16才起作用
    attrs["caffeSemantics"] = False

    # 001 is the default plugin version the parser will search for, and therefore can be omitted,
    # but we include it here for illustrative purposes.
    attrs["plugin_version"] = "1"

    return attrs

if __name__ == "__main__":
    onnx_path = "./end2end_folded_sim.onnx"
    graph = gs.import_onnx(onnx.load(onnx_path))

    # 添加op得到Efficient NMS plugin的input
    graph = gather_concat_score(graph)

    # 添加Efficient NMS plugin
    graph = create_and_add_plugin_node(graph, 20)

    # 保存图结构
    onnx.save(gs.export_onnx(graph),"./end2end_folded_sim_nms.onnx")







import tensorrt as trt
import numpy as np
import cuda


class ONNXClassifierWrapper():
    def __init__(self, file, num_classes, target_dtype = np.float32):

        self.target_dtype = target_dtype
        self.num_classes = num_classes
        self.load(file)

        self.stream = None

    def load(self, file):
        f = open(file, "rb")
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))

        engine = runtime.deserialize_cuda_engine(f.read())
        self.context = engine.create_execution_context()

    def allocate_memory(self, batch):
        self.output = np.empty(self.num_classes, dtype = self.target_dtype) # Need to set both input and output precisions to FP16 to fully enable FP16

        # Allocate device memory
        self.d_input = cuda.mem_alloc(1 * batch.nbytes)
        self.d_output = cuda.mem_alloc(1 * self.output.nbytes)

        self.bindings = [int(self.d_input), int(self.d_output)]

        self.stream = cuda.Stream()

    def predict(self, batch): # result gets copied into output
        if self.stream is None:
            self.allocate_memory(batch)

        # Transfer input data to device
        cuda.memcpy_htod_async(self.d_input, batch, self.stream)
        # Execute model
        self.context.execute_async_v2(self.bindings, self.stream.handle, None)
        # Transfer predictions back
        cuda.memcpy_dtoh_async(self.output, self.d_output, self.stream)
        # Syncronize threads
        self.stream.synchronize()

        return self.output



def convert_onnx_to_engine(onnx_filename, engine_filename = None, max_batch_size = 32, max_workspace_size = 1 << 30, fp16_mode = True):
    logger = trt.Logger(trt.Logger.WARNING)
    with trt.Builder(logger) as builder, builder.create_network() as network, trt.OnnxParser(network, logger) as parser:
        # builder.max_workspace_size = max_workspace_size
        # builder.fp16_mode = fp16_mode
        # builder.max_batch_size = max_batch_size

        print("Parsing ONNX file.")
        with open(onnx_filename, 'rb') as model:
            if not parser.parse(model.read()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))

        print("Building TensorRT engine. This may take a few minutes.")
        engine = builder.build_cuda_engine(network)

        if engine_filename:
            with open(engine_filename, 'wb') as f:
                f.write(engine.serialize())

        return engine, logger


print(trt.__version__)
PRECISION = np.float32

BATCH_SIZE = 64
N_CLASSES = 1000 # Our ResNet-50 is trained on a 1000 class ImageNet task

# convert_onnx_to_engine(onnx_filename='./resnet50/model.onnx', engine_filename="./resnet50/resnet_engine.trt",
#                        max_batch_size=BATCH_SIZE, max_workspace_size=1<<30, fp16_mode=False)

trt_model = ONNXClassifierWrapper("resnet_engine.trt", [BATCH_SIZE, N_CLASSES], target_dtype = PRECISION)
BATCH_SIZE=32
dummy_input_batch = np.zeros((BATCH_SIZE, 224, 224, 3), dtype = PRECISION)

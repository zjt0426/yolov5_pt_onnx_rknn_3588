if __name__ == '__main__':
    from rknn.api import RKNN

    # Create RKNN object
    rknn = RKNN(verbose=False)

    # Pre-process config
    print('--> Config model')
    rknn.config(mean_values=[[0, 0, 0]], std_values=[
                    [255, 255, 255]], target_platform='rk3588')
    print('done')

    # Load model
    print('--> Loading model')
    ret = rknn.load_onnx(model='yolov5s.onnx')
    #ret = rknn.load_pytorch(model=MODEL_PATH, input_size_list=[[1, 3, 640, 640]])
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=True, dataset='./dataset.txt')
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export rknn model
    print('--> Export rknn model')
    ret = rknn.export_rknn('yolov5s.rknn')
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')


    #print('--> Accuracy analysis')
    #ret = rknn.accuracy_analysis(inputs=['./subset/000000052891.jpg'])
    #if ret != 0:
    #    print('Accuracy analysis failed!')
    #    exit(ret)
    #print('done')

    # Release
    rknn.release()

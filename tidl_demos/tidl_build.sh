SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
ORT_ROOT=$SCRIPT_DIR/..

cd $ORT_ROOT

./build.sh --config RelWithDebInfo --build_shared_lib --parallel --skip_tests --skip_onnx_tests --use_tidl --use_dnnl
#./build.sh --config Debug --build_shared_lib --parallel --skip_tests --skip_onnx_tests --use_tidl --use_dnnl

cd -

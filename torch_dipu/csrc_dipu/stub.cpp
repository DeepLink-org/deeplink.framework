#include <csrc_dipu/binding/exportapi.h>

class Initializer final {
public:
    Initializer() {
    }
    Initializer(const Initializer&) = delete;
    Initializer& operator=(const Initializer&) = delete;
    ~Initializer() {
    }
};
static Initializer init;

static std::vector<PyMethodDef> methods;

static void AddPyMethodDefs(std::vector<PyMethodDef>& vector, PyMethodDef* methods)
{
  if (!vector.empty()) {
    // remove nullptr terminator
    vector.pop_back();
  }
  while (true) {
    vector.push_back(*methods);
    if (!methods->ml_name) {
      break;
    }
    methods++;
  }
}

extern "C" PyObject* initModule() {

  AddPyMethodDefs(methods, dipu::exportTensorFunctions());
  static struct PyModuleDef torchnpu_module = {
     PyModuleDef_HEAD_INIT,
     "torch_dipu._C",
     nullptr,
     -1,
     methods.data()
  };
  PyObject* module = PyModule_Create(&torchnpu_module);

  dipu::exportDIPURuntime(module);
  return module;
}

PyMODINIT_FUNC PyInit__C(void){
  return initModule();
}



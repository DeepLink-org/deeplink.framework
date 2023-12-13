import os
import yaml

accepted_interface = "ALL"

class OpMemoryFormatConverter(object):
    def __init__(self, convert_config):
        assert(isinstance(convert_config, str))
        if convert_config and len(convert_config):
            with open(convert_config) as convert_config_yaml_file:
                file_data = convert_config_yaml_file.read()
                self.convert_config_yaml = yaml.load(file_data, Loader=yaml.FullLoader)
                self.convert_config = ConvertConfig(self.convert_config_yaml)
        else:
            self.convert_config_yaml = list()
            self.convert_config = ConvertConfig(self.convert_config_yaml)

    def convert(self,custom_code,fun_config):
        if "interface" in fun_config and (accepted_interface == "ALL" or (fun_config['interface'] in accepted_interface)):
            return self.do_convert(custom_code,fun_config)
        else:
            return custom_code
    
    def do_convert(self,custom_code,fun_config):
        interface = fun_config['interface']
        memory_format = self.convert_config.interface2memoryformat(interface)
        if memory_format == "channellast":
            custom_code = custom_code.replace("$SUGGESTED_MEMORYFORMAT_3D","at::MemoryFormat::ChannelsLast3d").replace("$SUGGESTED_MEMORYFORMAT","at::MemoryFormat::ChannelsLast")
        elif memory_format == "contiguous":
            custom_code = custom_code.replace("$SUGGESTED_MEMORYFORMAT_3D","at::MemoryFormat::Contiguous").replace("$SUGGESTED_MEMORYFORMAT","at::MemoryFormat::Contiguous")
        elif memory_format == "preserve":
            custom_code = custom_code.replace("$SUGGESTED_MEMORYFORMAT_3D","at::MemoryFormat::Preserve").replace("$SUGGESTED_MEMORYFORMAT","at::MemoryFormat::Preserve")
        elif memory_format == "empty":
            while " $SUGGESTED_MEMORYFORMAT" in custom_code:
                custom_code = custom_code.replace(" $SUGGESTED_MEMORYFORMAT","$SUGGESTED_MEMORYFORMAT")
            custom_code = custom_code.replace(",$SUGGESTED_MEMORYFORMAT_3D","").replace(",$SUGGESTED_MEMORYFORMAT","")
        else:
            print("UNABLE TO RECOGNIZE MEMORY FORMAT!!!")
        return custom_code

class ConvertConfig(object):
    def __init__(self, config_yaml):
        self.convert_dict = dict()
        self.convert_config_yaml = config_yaml
        self.default_layout = "empty"
        assert(isinstance(config_yaml, list))
        for config in config_yaml:
            assert(isinstance(config,dict))
            for interface in config.keys():
                if interface == "common_config":
                    detail = config[interface]
                    assert(isinstance(detail, dict))
                    if "layout" in detail:
                        self.default_layout = self.layout2memoryformat(detail["layout"])
                    pass
                    # may add common behavior
            for interface in config.keys():
                if interface != "common_config":
                    self.convert_dict.setdefault(interface,dict())
                    detail = config[interface]
                    assert(isinstance(detail, dict))
                    if "layout" in detail:
                        self.convert_dict[interface]["layout"] = self.layout2memoryformat(detail["layout"])
 
    def layout2memoryformat(self, layout):
        assert(isinstance(layout, str))
        if "NCHW" in layout:
            return "contiguous"
        if "NLC" in layout:
            return "channellast"
        if "NHWC" in layout:
            return "channellast"
        if "NDHWC" in layout:
            return "channellast"
        return "preserve"
     
    def interface2memoryformat(self, interface):
        interface_stripped = interface.strip().split("(")[0]
        if (interface_stripped not in self.convert_dict) or ("layout" not in self.convert_dict[interface_stripped]):
            return self.default_layout
        else:
            return self.convert_dict[interface_stripped]["layout"]

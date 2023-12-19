import os
import re
import yaml

accepted_interface = "ALL"

class OpMemoryFormatConverter(object):
    #The converter class, will do the converting memory format based on the convert_config.yaml loaded.
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
        # Do the covert job
        def choose_default(matched):
            value = str(matched.group("default"))
            return value
        
        def choose_channelsLast3d(matched):
            return "at::MemoryFormat::ChannelsLast3d"
        
        def choose_channelsLast(matched):
            return "at::MemoryFormat::ChannelsLast"
        
        def choose_contiguous(matched):
            return "at::MemoryFormat::Contiguous"

        def choose_preserve(matched):
            return "at::MemoryFormat::Preserve"

        interface = fun_config["interface"]
        custom_code = custom_code.split("\n")
        memory_format = self.convert_config.interface2memoryformat(interface)
        custom_code_new = list()
        # match string like "${PREFERRED_MEMORY_FORMAT_PLACHOLDER_3D:-<default>}"
        placeholder_3d_pattern = "\$\{PREFERRED_MEMORY_FORMAT_PLACEHOLDER_3D:-(?P<default>.*)\}"
        # match string like "${PREFERRED_MEMORY_FORMAT_PLACHOLDER:-<default>}"
        placeholder_pattern = "\$\{PREFERRED_MEMORY_FORMAT_PLACEHOLDER:-(?P<default>.*)\}"
        for line in custom_code:
            if memory_format == "channellast":
                line = re.sub(placeholder_3d_pattern, choose_channelsLast3d, line)
                line = re.sub(placeholder_pattern, choose_channelsLast, line)
            elif memory_format == "contiguous":
                line = re.sub(placeholder_3d_pattern, choose_contiguous, line)
                line = re.sub(placeholder_pattern, choose_contiguous, line)
            elif memory_format == "preserve":
                line = re.sub(placeholder_3d_pattern, choose_preserve, line)
                line = re.sub(placeholder_pattern, choose_preserve, line)
            elif memory_format == "empty":
                line = re.sub(placeholder_3d_pattern, choose_default, line)
                line = re.sub(placeholder_pattern, choose_default, line)
            else:
                print("UNABLE TO RECOGNIZE MEMORY FORMAT!!!")
            custom_code_new.append(line)
        custom_code = "\n".join(custom_code_new)
        return custom_code

class ConvertConfig(object):
    #This class is used to load and parse the convert_config.yaml
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
        #used when pasing convert_config.yaml, return the memory format based on NCHW/NHWC and other layout.
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
        #return the prefered memory format based on the DIOPI interface.
        interface_stripped = interface.strip().split("(")[0]
        if (interface_stripped not in self.convert_dict) or ("layout" not in self.convert_dict[interface_stripped]):
            return self.default_layout
        else:
            return self.convert_dict[interface_stripped]["layout"]

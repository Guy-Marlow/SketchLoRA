def get_model(model_name, args):
    name = model_name.lower()
    if name == "ease":
        from models.ease import Learner
    elif name == 'tuna':
        from models.tuna import Learner
    elif name == 'lora':
        from models.lora import Learner
    elif name == 'seqlora':
        from models.seqlora import Learner
    elif name == 'sdlora':
        from models.sdlora import Learner
    elif name == 'sdlora_sketch':
        from models.sdlora_sketch import Learner
    elif name == 'olora':
        from models.olora import Learner
    elif name == 'inflora':
        from models.inflora import Learner
    elif name == 'lorasub':
        from models.lorasub import Learner
    elif name == 'lorasub_sketch':
        from models.lorasub_sketch import Learner
    elif name == 'sketchlora_batchinc':
        from models.sketchlora_batchinc import Learner
    elif name == 'sketchlora':
        from models.sketchlora import Learner
    elif name == 'sketchlora_align':
        from models.sketchlora_align import Learner
    elif name == 'cllora':
        from models.cllora import Learner
    elif name == 'rainbowprompt':
        from models.rainbowprompt import Learner
    else:
        assert 0
    return Learner(args)

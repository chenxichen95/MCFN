import torch
import os


def load_state_dict(model, args, running_time, baseline):
    log_dir_name = running_time
    save_dir = os.path.join(args.output_dir, f'{baseline}', log_dir_name)  + f'/pytorch_model.bin'
    state_dict = torch.load(save_dir, map_location='cpu')
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    # new_state_dict=state_dict.copy()
    # for kye ,value in state_dict.items():
    #     new_state_dict[kye.replace("bert","c_bert")]=value
    # state_dict=new_state_dict
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})

        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            # logger.info("name {} chile {}".format(name,child))
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix='' if hasattr(model, 'bert') else 'bert.')
    print("missing keys:{}".format(missing_keys))
    print('unexpected keys:{}'.format(unexpected_keys))
    print('error msgs:{}'.format(error_msgs))

    return model


def load_best_model(model, args, running_time, baseline):
    log_dir_name = running_time
    save_dir = os.path.join(args.output_dir, f'{baseline}', log_dir_name) + f'/pytorch_model.bin'
    state_dict = torch.load(save_dir, map_location='cpu')
    state_dict2 = {}
    for key, value in state_dict.items():
        state_dict2[f'module.{key}'] = value
    model.load_state_dict(state_dict2)

    return model


def save_model(model, global_step, args, logging, log, running_time, baseline):
    log_dir_name = running_time
    save_dir = os.path.join(args.output_dir, f'{baseline}', log_dir_name)
    if os.path.exists(save_dir) == False:
        os.makedirs(save_dir, exist_ok=True)

    # save file
    with open(save_dir + '/{}'.format('config.py'), 'w', encoding='utf-8') as fp1:
        with open(f'{args.root_dir}/config.py', 'r', encoding='utf-8') as fp2:
            fp1.write(fp2.read())
    if os.path.exists(f'{args.root_dir}/train.log'):
        with open(save_dir + '/{}'.format('train.log'), 'w', encoding='utf-8') as fp1:
            with open(f'{args.root_dir}/train.log', 'r', encoding='utf-8') as fp2:
                fp1.write(fp2.read())

    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    logging.info(f'save model: {save_dir}/pytorch_model.bin')
    log.print(f'save model: {save_dir}/pytorch_model.bin')
    torch.save(model_to_save.state_dict(), save_dir + f'/pytorch_model.bin')

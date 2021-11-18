if __name__ == '__main__':
    # in this way by judging the mark of args, users will decide which function to use
    if not args_opt.do_eval and args_opt.run_distribute:
        context.set_auto_parallel_context(device_num=args_opt.device_num, parallel_mode=ParallelMode.AUTO_PARALLEL,
                                          all_reduce_fusion_config=[140])
        init()

    epoch_size = args_opt.epoch_size
    net = resnet50(args_opt.batch_size, args_opt.num_classes)
    ls = SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    opt = Adam(filter(lambda x: x.requires_grad, net.get_parameters()), 0.01, 0.9)
    model = Model(net, loss_fn=ls, optimizer=opt, metrics={'acc'})

    # as for train, users could use model.train
    if args_opt.do_train:
        datasets = create_dataset()
        sizes = [1/kfold_num]*kfold_num
        groups = datasets.split(sizes, randomize=True)
        batch_num = datasets.get_dataset_size()
        validation_res_list = []
        for fold in range(kfold_num):
            config_ck = CheckpointConfig(save_checkpoint_steps=batch_num*EPOCH_PER_CKPT, keep_checkpoint_max=30)
            ckpoint_cb = ModelCheckpoint(prefix="train_resnet_distribute", directory="./distribute_train", config=config_ck)
            loss_cb = LossMonitor()
            validation_data = None
            for group in groups[fold:(fold+1)]:
                if validation_data is None:
                    validation_data = group
                else:
                    validation_data = validation_data + group
            training_data = None
            for group in groups[:fold] + groups[(fold+1):]:
                if training_data is None:
                    training_data = group
                else:
                    training_data = training_data + group
            model.train(epoch_size, training_data, callbacks=[ckpoint_cb, loss_cb])
            training_res = model.eval(training_data)
            validation_res = model.eval(validation_data)
            print("training result: ", training_res)
            validation_res_list.append(validation_res_list)
        validation_res = np.average(validation_res)
        print("validation result: ", validation_res)

    # as for evaluation, users could use model.eval
    if args_opt.do_eval:
        if args_opt.checkpoint_path:
            param_dict = load_checkpoint(args_opt.checkpoint_path)
            load_param_into_net(net, param_dict)
        eval_dataset = create_dataset(training=False)
        res = model.eval(eval_dataset)
        print("result: ", res)
        logging.info("result: ", res)

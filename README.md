# BestCheckpointCopier

使用tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)训练模型，设置epoch后，训练能够报错的模型只是最后几轮的checkpoints，而不是保存效果最佳的checkpoints，比如loss最小或者acc最大等等，实现这个exporter就是能够满足保存最佳模型的需求。

使用方式如下：
     best_exporter = best_checkpoint_copier.BestCheckpointCopier(
                        name='best', # directory within model directory to copy checkpoints to
                        checkpoints_to_keep=5, # number of checkpoints to keep
                        score_metric='loss', # eval_result metric to use to determine "best"
                        compare_fn=lambda x,y: x.score < y.score, # comparison function used to determine "best" checkpoint (x is the current checkpoint; y is the previously copied checkpoint with the highest/worst score)
                        sort_key_fn=lambda x: x.score, # key to sort on when discarding excess checkpoints
                        sort_reverse=False) # sort order when discarding excess checkpoints
    exporters = [best_exporter]

    train_spec = tf.estimator.TrainSpec(
            input_fn=self.input_fn.train_input_fn,
            max_steps=self.input_fn.num_train_steps,
            #hooks=[early_stopping_hook]
        )

    eval_spec = tf.estimator.EvalSpec(
            input_fn=self.input_fn.eval_input_fn,
            exporters=exporters,
            throttle_secs=200
        )

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


#coding=utf-8

"""
作者：welliechen
日期：2020-06-06
功能：保存最优指标对应模型
"""

import glob
import os
import shutil
import tensorflow as tf
import json


'''
定义checkpoint结构
dir
file
score
path
'''
class Checkpoint(object):
  dir = None
  file = None
  score = None
  path = None

  def __init__(self, path, score):
    self.dir = os.path.dirname(path)
    self.file = os.path.basename(path)
    self.score = score
    self.path = path


'''
继承tf.estimator.Exporter
'''
class BestCheckpointCopier(tf.estimator.Exporter):
  checkpoints = None
  checkpoints_to_keep = None
  compare_fn = None
  name = None
  score_metric = None
  sort_key_fn = None
  sort_reverse = None

  def __init__(self, name='best_checkpoints', checkpoints_to_keep=5, score_metric='Loss/total_loss', compare_fn=lambda x,y: x.score < y.score, sort_key_fn=lambda x: x.score, sort_reverse=False):
    self.checkpoints = []
    self.checkpoints_to_keep = checkpoints_to_keep
    self.compare_fn = compare_fn
    self.name = name
    self.score_metric = score_metric
    self.sort_key_fn = sort_key_fn
    self.sort_reverse = sort_reverse
    super(BestCheckpointCopier, self).__init__()
  
  '''
  创建目标路径，并拷贝best checkpoints
  '''
  def _copyCheckpoint(self, checkpoint):
    desination_dir = self._destinationDir(checkpoint)
    os.makedirs(desination_dir, exist_ok=True)

    for file in glob.glob(r'{}*'.format(checkpoint.path)):
      self._log('copying {} to {}'.format(file, desination_dir))
      shutil.copy(file, desination_dir)

  def _destinationDir(self, checkpoint):
    return os.path.join(checkpoint.dir, self.name)

  def _keepCheckpoint(self, checkpoint):
    self._log('keeping checkpoint {} with score {}'.format(checkpoint.file, checkpoint.score))

    self.checkpoints.append(checkpoint)
    self.checkpoints = sorted(self.checkpoints, key=self.sort_key_fn, reverse=self.sort_reverse)

    self._copyCheckpoint(checkpoint)

  def _log(self, statement):
    tf.logging.info('[{}] {}'.format(self.__class__.__name__, statement))

  '''
  根据score调整best checkpoints，删除不满足的checkpoints
  '''
  def _pruneCheckpoints(self, checkpoint):
    destination_dir = self._destinationDir(checkpoint)

    for checkpoint in self.checkpoints[self.checkpoints_to_keep:]:
      self._log('removing old checkpoint {} with score {}'.format(checkpoint.file, checkpoint.score))

      old_checkpoint_path = os.path.join(destination_dir, checkpoint.file)
      for file in glob.glob(r'{}*'.format(old_checkpoint_path)):
        self._log('removing old checkpoint file {}'.format(file))
        os.remove(file)

    self.checkpoints = self.checkpoints[0:self.checkpoints_to_keep]

  def _score(self, eval_result):
    return float(eval_result[self.score_metric])

  def _shouldKeep(self, checkpoint):
    return len(self.checkpoints) < self.checkpoints_to_keep or self.compare_fn(checkpoint, self.checkpoints[-1])

  '''
  实现export方法
  '''
  def export(self, estimator, export_path, checkpoint_path, eval_result, is_the_final_export):
    self._log('export checkpoint {}'.format(checkpoint_path))
    self._log('eval_result {}'.format(str(eval_result)))

    score = self._score(eval_result)
    checkpoint = Checkpoint(path=checkpoint_path, score=score)

    if self._shouldKeep(checkpoint):
      self._keepCheckpoint(checkpoint)
      self._pruneCheckpoints(checkpoint)
    else:
      self._log('skipping checkpoint {}'.format(checkpoint.path))

    # 判断是否为最后一次导出，如果是，就输出整体评估结果
    if is_the_final_export:
        info = dict()
        for checkpoint in self.checkpoints:
            info[checkpoint.file] = checkpoint.score
        best_chkpt_dir = os.path.join(checkpoint.dir, self.name)
        eval_result_file = os.path.join(best_chkpt_dir, "eval_best_results.json")
        with open(eval_result_file, 'w') as f:
            json.dump(info, f, indent=4)



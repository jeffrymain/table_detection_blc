import imp
from .eval_hooks import EvalHook

class MyEvalHook(EvalHook):
    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        if not self._should_evaluate(runner):
            return

        from mmdet.apis import single_gpu_test, my_single_gpu_test
        
        # results = single_gpu_test(runner.model, self.dataloader, show=False)
        results, line_results = my_single_gpu_test(runner.model, self.dataloader, show=False)


        # 进行评价
        runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)

        line_score = self.dataloader.dataset.evaluate_lines(runner, results=line_results)

        key_score = self.evaluate(runner, results)
        # the key_score may be `None` so it needs to skip the action to save
        # the best checkpoint
        if self.save_best and key_score:
            self._save_ckpt(runner, key_score)
        


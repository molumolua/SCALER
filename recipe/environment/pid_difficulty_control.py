import math, random
import json
class PIDDifficultyControl:
    def __init__(self, target=0.5,k=2,dmin=0.0, dmax=50.0, step_cap=0.5, jitter=0.15,state = None,activate_function="linear",ema_list=None):
        self.target = target
        self.k=k
        self.dmin, self.dmax = dmin, dmax
        self.step_cap = step_cap
        self.jitter = jitter
        
        self.activate_function =activate_function
        
        if state:
            self.state =state
        else:
            self.state = {
                "d": 0.0, "t": 0
            } 
        
        if ema_list:
            self.ema_list = ema_list
        else:
            self.ema_list = [None for _ in range(dmax-dmin+1)]


    def propose_distances(self,batch_size):
        """为一个批次生成 per-instance 的 distance 列表（含混合 & 轻微抖动）。"""
        st = self.state
        d = st["d"]
        # 用 floor/ceil 混合来表达非整数 difficulty（见第 4 节）
        lo, hi = math.floor(d), math.ceil(d)
        frac = d - lo
        num_hi = round(batch_size * frac)
        distances = [hi]*num_hi + [lo]*(batch_size - num_hi)
        return distances

    def update(self,distance,distance_correct_avg_dict):
        """用一个批次的正确率更新控制器，并返回新的 d。"""
        st = self.state
        
        for distance,correct in distance_correct_avg_dict:
            distance
        err = batch_correct_ratio - self.target
            
        raw_delta = self.k*err
        
        
        if self.activate_function =="linear":
            delta = raw_delta
                # 限幅，防止抖动/超调
            if delta > self.step_cap: delta = self.step_cap
            if delta < -self.step_cap: delta = -self.step_cap
        elif self.activate_function == "tanh":
            delta = math.tanh(3*raw_delta)/math.tanh(3) * self.step_cap
        else:
            raise NotImplementedError
            
        # 限幅，防止抖动/超调
        if delta > self.step_cap: delta = self.step_cap
        if delta < -self.step_cap: delta = -self.step_cap
        

        st["d"] = max(self.dmin, min(self.dmax, st["d"] + delta))
        st["t"] += 1
        
    
    
    def _to_serializable(self):
        """转成可 JSON 的纯基础类型结构。注意 defaultdict 不能直接序列化，需要转成普通 dict。"""
        return {
            "version": 1,
            "params": {
                "target": self.target,
                "k": self.k,
                "dmin": self.dmin, "dmax": self.dmax,
                "step_cap": self.step_cap,
                "jitter": self.jitter,
                "state": {k: v for k, v in self.state.items()},
                "activate_function":self.activate_function
            },
        }

    @classmethod
    def _from_serializable(cls, payload):
        """从 JSON 结构恢复对象"""
        params = payload["params"]
        obj = cls(**params)
        return obj

    @staticmethod
    def json_default(o):
        """给 json.dump 用：遇到 DifficultyControl 就转 dict。"""
        if isinstance(o, DifficultyControl):
            return o._to_serializable()
        # 交回给 json 触发 TypeError（能更早发现其它不可序列化对象）
        raise TypeError(f"{type(o)} is not JSON serializable")

    @staticmethod
    def json_object_hook(d):
        """给 json.load 用：遇到我们写出的结构就还原成 DifficultyControl 对象。"""
        if isinstance(d, dict) and d.get("version") == 1 and "params" in d:
            try:
                return DifficultyControl._from_serializable(d)
            except Exception:
                # 失败就原样返回，避免 load 整体失败
                return d
        return d


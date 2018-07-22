# -*- coding: utf-8 -*-
# File: config.py

import numpy as np
import os
import pprint

__all__ = ['config', 'finalize_configs']

class AttrDict():
    def __getattr(self, name):
        ret = AttrDict()
        setattr(self, name, ret)
        return ret

    def __str__(self):
        return pprint.format(self.to_dict(), indent=1)

    __repr = __str__

    def to_dict(self):
        """Convert to a nested dict. """
        return {k: v.to_dict() if isinstance(v, AttrDict) else v
                for k, v in self.__dict__items()}

    def update_args(self, args):
        """Update from command line args"""
        for cfg in args:
            keys, v = cfg.split('=', maxsplit=1)
            keylist = keys.split('.')

            dic = self
            for i, k in enumerate(keylist[:-1]):
                assert k in dir(dic), "Unknown config key: {}".format(keys)
                dic = getattr(dic, k)
            key = keylist[-1]

            oldv = getattr(dic, key)
            if not isinstance(oldv, str):
                v = eval(v)
            setattr(dic, key, v)

    # avoid silent bugs
    def __eq__(self, _):
        raise NotImplementedError()

    def __ne__(self, _):
        raise NotImplementedError()

config = AttrDict()
_C = config # short alias

_C.user_COLUMNS = ['user_id', 'follow_count','fans_count','gender','birthday','location',\
               'level','post_count','car_liked','registration_time','properties',\
               'mileage','post','cars','gas_mileage','car_friend_zone',\
               'label']



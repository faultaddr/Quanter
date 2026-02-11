class QAdapter(object):
    """
    Adapts an object by replacing methods.
    Usage:
    aObj = A()
    aObj = Adapter(aObj, make_compute=aObj.compute)
    bObj = B()
    bObj = Adapter(bObj, make_compute=Obj.b_compute)
    """

    def __init__(self, obj, **adapted_methods):
        """We set the adapted methods in the object's dict"""
        self.obj = obj
        self.__dict__.update(adapted_methods)

    def __getattr__(self, attr):
        """All non-adapted calls are passed to the object"""
        return getattr(self.obj, attr)

    def original_dict(self):
        """Print original object dict"""
        return self.obj.__dict__


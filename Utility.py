import pickle
import os

class LoadSavePickle:
    # extension
    ext = ".pickle"

    def dump(self, filename, obj):
        filename = filename + self.ext
        with open(filename, 'w') as f:
            pickle.dump(obj, f)
            f.close()

    def load(self, filename):
        retval = None
        filename = filename + self.ext
        if os.path.isfile(filename):
            with open(filename, 'r') as f:
                retval = pickle.load(f)
                f.close()
        else:
            print "Pickle error: %s not found" % filename
        return retval

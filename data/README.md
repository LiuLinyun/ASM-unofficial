download hifi3d++.mat here.

## data files structure

### hifi3dpp.mat


### weights_map.pkl
weights_map.pkl is an python object dict dumped by `pickle` like:

~~~python
{
  "<bone_name_1>" : {
    "vertex_indices": np.array(some_value), # i-th element represents i-th vertex of mesh is(True) or not(False) weighted by this bone named bone_name_1
    "weights": np.array(some_value) # i-th element represents the weight of this bone on the i-th vertex 
  },
  "<bone_name_2>" : {
    "vertex_indices": np.array(some_value),
    "weights": np.array(some_value)
  }
  # ... other bones' weights
}
~~~
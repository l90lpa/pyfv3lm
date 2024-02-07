# PyFV3LM

## Development

### Questions:
#### fv_grid_bounds_type:
- [ ] should the following always hold true for the parameters of fv_dynamics `npx == bd%ie - bd%is + 1` etc?
- [ ] should the following always be true, `is == isc` etc? It seems to be the case from looking at how fv_mp_mod::domain_decomp creates the grid bounds.

#### c2l_ord4:
- In the fortran version, we have,
  ```
    real utmp(bd%is:bd%ie,  bd%js:bd%je+1)
    real vtmp(bd%is:bd%ie+1,bd%js:bd%je)
  ```
  However, it seems that the reads and writes of `utmp` and `vtmp` are only ever performed in the range `bd%is:bd%ie, bd%js:bd%je`. Thus I believe we could reduce there storage to reflect this?

## Documentation

### Grid
In GFDL's Fortran based implementation of FV3, array indexing on a sub-tile uses a 1-based tile local index space, while in PyFV3LM we use a 0-based sub-tile local index space. Additionally in PyFV3LM we set grid bounds, ie, iec, ied, etc to one past the end, as opposed to in the Fortran based implementation where these are equal to the index of the last element, however, because of the difference in base index this often doesn't appear explicitly. 

#### Tile Local vs Sub Tile Local Example:
Suppose that we have a tile of size 2N partitioned into 2 sub-tiles on the I-axis (x-axis), with a halo size of 3. then the difference in index space is as follows:

|Index space| Sub Tile 1 | Sub Tile 2|
| --- | --- | --- |
|Tile Local | isd = -2, ied = N+3 | isd = N, ied = 2N+3 |
|Sub Tile Local | isd = 0, ied = N+6 | isd = 0, ied = N+6 |
|Tile Local | isc = 1, ied = N | isd = N+1, ied = 2N |
|Sub Tile Local | isd = 3, ied = N+3 | isd = 3, ied = N+3 |
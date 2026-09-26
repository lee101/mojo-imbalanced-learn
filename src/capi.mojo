"""Dense resampling kernels exposed through a small C ABI."""

from std.sys.info import simd_width_of

comptime Ptr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = UnsafePointer[Int64, AnyOrigin[mut=True]]
comptime W = simd_width_of[DType.float64]()
comptime MAX_INDEX = 9223372036854775807


def p(addr: Int) -> Ptr:
    return Ptr(unsafe_from_address=addr)


def ip(addr: Int) -> IPtr:
    return IPtr(unsafe_from_address=addr)


def squared_distance(a: Ptr, b: Ptr, d: Int) -> Float64:
    var acc = SIMD[DType.float64, W](0.0)
    var j = 0
    while j + W <= d:
        var delta = a.load[width=W](j) - b.load[width=W](j)
        acc += delta * delta
        j += W
    var total = acc.reduce_add()
    while j < d:
        var delta = a[j] - b[j]
        total += delta * delta
        j += 1
    return total


def smote_neighbor_rows(
    x: Ptr,
    indices: IPtr,
    distances: Ptr,
    n: Int,
    d: Int,
    width: Int,
    begin: Int,
    end: Int,
):
    for query in range(begin, end):
        var base = query * width
        for slot in range(width):
            distances[base + slot] = 1.7976931348623157e308
            indices[base + slot] = -1
        for row in range(n):
            var distance = squared_distance(x + query * d, x + row * d, d)
            if distance >= distances[base + width - 1]:
                continue
            var slot = width - 1
            while slot > 0 and distances[base + slot - 1] > distance:
                distances[base + slot] = distances[base + slot - 1]
                indices[base + slot] = indices[base + slot - 1]
                slot -= 1
            distances[base + slot] = distance
            indices[base + slot] = Int64(row)


@export("mil_smote_neighbors")
def mil_smote_neighbors(
    x_addr: Int,
    index_addr: Int,
    distance_addr: Int,
    n: Int,
    d: Int,
    k: Int,
) abi("C") -> Int32:
    """Return k+1 sorted neighbors per row, including the query point."""
    if x_addr == 0 or index_addr == 0 or distance_addr == 0:
        return 1
    if n <= 0 or d <= 0 or k <= 0 or k >= n:
        return 2
    if n > MAX_INDEX // d or n > MAX_INDEX // (k + 1):
        return 2
    smote_neighbor_rows(
        p(x_addr), ip(index_addr), p(distance_addr), n, d, k + 1, 0, n
    )
    return 0


@export("mil_smote_neighbors_range")
def mil_smote_neighbors_range(
    x_addr: Int,
    index_addr: Int,
    distance_addr: Int,
    n: Int,
    d: Int,
    width: Int,
    begin: Int,
    end: Int,
) abi("C") -> Int32:
    """Fill rows [begin, end) of the neighbor table; the shim fans this out."""
    if x_addr == 0 or index_addr == 0 or distance_addr == 0:
        return 1
    if (
        n <= 0
        or d <= 0
        or width <= 0
        or width > n
        or n > MAX_INDEX // d
        or n > MAX_INDEX // width
    ):
        return 2
    if begin < 0 or end > n or begin >= end:
        return 2
    smote_neighbor_rows(
        p(x_addr), ip(index_addr), p(distance_addr), n, d, width, begin, end
    )
    return 0


@export("mil_smote_generate")
def mil_smote_generate(
    x_addr: Int,
    neighbor_addr: Int,
    selection_addr: Int,
    step_addr: Int,
    dst_addr: Int,
    n: Int,
    d: Int,
    k: Int,
    m: Int,
) abi("C") -> Int32:
    """Generate samples from flattened (base row, neighbor slot) choices."""
    if (
        x_addr == 0
        or neighbor_addr == 0
        or selection_addr == 0
        or step_addr == 0
        or dst_addr == 0
    ):
        return 1
    if n <= 0 or d <= 0 or k <= 0 or k >= n or m < 0:
        return 2
    if n > MAX_INDEX // d or m > MAX_INDEX // d:
        return 2
    var x = p(x_addr)
    var neighbors = ip(neighbor_addr)
    var selections = ip(selection_addr)
    var steps = p(step_addr)
    var dst = p(dst_addr)
    var neighbor_width = k + 1
    for sample in range(m):
        var selection = Int(selections[sample])
        if selection < 0 or selection // k >= n:
            return 3
        var row = selection // k
        var slot = selection - row * k
        var neighbor = Int(neighbors[row * neighbor_width + slot + 1])
        if neighbor < 0 or neighbor >= n:
            return 3
        var alpha = SIMD[DType.float64, W](steps[sample])
        var j = 0
        while j + W <= d:
            var a = x.load[width=W](row * d + j)
            var b = x.load[width=W](neighbor * d + j)
            dst.store(sample * d + j, a + alpha * (b - a))
            j += W
        while j < d:
            var a = x[row * d + j]
            dst[sample * d + j] = a + steps[sample] * (x[neighbor * d + j] - a)
            j += 1
    return 0


def gather_rows(x: Ptr, indices: IPtr, dst: Ptr, d: Int, rows: Int):
    for target in range(rows):
        var source = Int(indices[target])
        var j = 0
        while j + W <= d:
            dst.store(target * d + j, x.load[width=W](source * d + j))
            j += W
        while j < d:
            dst[target * d + j] = x[source * d + j]
            j += 1


@export("mil_gather_f64")
def mil_gather_f64(
    x_addr: Int,
    index_addr: Int,
    dst_addr: Int,
    source_rows: Int,
    rows: Int,
    d: Int,
) abi("C") -> Int32:
    if x_addr == 0 or index_addr == 0 or dst_addr == 0:
        return 1
    if source_rows <= 0 or rows < 0 or d <= 0:
        return 2
    if source_rows > MAX_INDEX // d or rows > MAX_INDEX // d:
        return 2
    var x = p(x_addr)
    var indices = ip(index_addr)
    var dst = p(dst_addr)
    for target in range(rows):
        var source = Int(indices[target])
        if source < 0 or source >= source_rows:
            return 3
    # A row gather moves d doubles per row and does no arithmetic worth
    # splitting, so it stays on one core regardless of size.
    gather_rows(x, indices, dst, d, rows)
    return 0

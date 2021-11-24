## Parallel Prefix Algorithm
 - binary operator but we want to apply it piecewise: a+b+c+...
 - assumption that it is associative, otherwise not really possible
 - any subsum should in the end be accessible, we should basically get a summed up array from an original array
 - generally $O(n)$ computation

### Naive parallal prefix algorithm
 - assumption that number of elements is power of 2 ($n = 2^k$)
 - gives $O(log(n))$
 - 
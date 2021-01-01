<span style="float:right;"><a href="https://github.com/RubixML/Extras/blob/master/src/Persisters/Serializers/Bzip2.php">[source]</a></span>

# Bzip2
A compression format based on the Burrows–Wheeler transform. Bzip2 is slightly smaller than Gzip format but is slower and requires more memory.

> **Note:** This serializer requires the Bzip2 PHP extension.

## Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | block size | 4 | int | The size of each block between 1 and 9 where 9 gives the best compression. |
| 2 | work factor | 0 | int | Controls how the compression phase behaves when the input is highly repetitive. |
| 3 | serializer | Native | Serializer | The base serializer |

## Example
```php
use Rubix\ML\Persisters\Serializers\Bzip2;
use Rubix\ML\Persisters\Serializers\Native;

$serializer = new Bzip2(4, 125, new Native());
```

### References
>- J. Tsai. (2006). Bzip2: Format Specification.
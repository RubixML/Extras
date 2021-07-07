<span style="float:right;"><a href="https://github.com/RubixML/Extras/blob/master/src/Tokenizers/KMer.php">[source]</a></span>

# K-mer
K-mers are substrings of sequences such as DNA containing the bases A, T, C, and G with a length of *k*. They are often used in bioinformatics to represent features of a DNA sequence.

!!! note
    K-mers that contain invalid bases will not be generated.

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | k | 4 | int | The length of tokenized sequences. |

## Example
```php
use Rubix\ML\Tokenizers\Whitespace;

$tokenizer = new KMer(4);
```

## Additional Methods
Return the number of k-mers that were dropped due to invalid bases.
```php
public function dropped() : int
```

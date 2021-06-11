<span style="float:right;"><a href="https://github.com/RubixML/Extras/blob/master/src/Extractors/Deduplicator.php">[source]</a></span>

# Deduplicator
Removes duplicate records from a dataset while the records are in flight. Deduplicator uses a Bloom filter under the hood to probabilistically identify records the filter has already seen before.

**Interfaces:** [Extractor](api.md)

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | iterator | | Traversable | The base iterator. |
| 2 | size | | int | The size of the bloom filter. |
| 3 | numHashes | 3 | int | The number of hash functions. |

## Example
```php
use Rubix\ML\Extractors\Deduplicator;
use Rubix\ML\Extractors\CSV;

$extractor = new Deduplicator(new CSV('example.csv', true), 100000, 3);
```

## Additional Methods
Return the probability of mistakenly dropping a unique record.
```php
public falsePositiveRate() : float
```

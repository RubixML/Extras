<span style="float:right;"><a href="https://github.com/RubixML/Extras/blob/master/src/Transformers/DeltaTfIdfTransformer.php">[source]</a></span>

# Delta TF-IDF Transformer
A supervised TF-IDF (Term Frequency Inverse Document Frequency) Transformer that uses class labels to boost the TF-IDFs of terms by how informative they are. Terms that receive the highest boost are those whose concentration is primarily in one class whereas low weighted terms are more evenly distributed among the classes.

> **Note:** Delta TF-IDF Transformer assumes that its inputs are token frequency vectors such as those created by [Word Count Vectorizer](word-count-vectorizer.md).

**Interfaces:** [Transformer](api.md#transformer), [Stateful](api.md#stateful), [Elastic](api.md#elastic)

**Data Type Compatibility:** Continuous only

## Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | smoothing | 1.0 | float | The amount of additive (Laplace) smoothing to add to the term frequencies and inverse document frequencies (IDFs). |

## Example
```php
use Rubix\ML\Transformers\DeltaTfIdfTransformer;

$transformer = new DeltaTfIdfTransformer(1.0);
```

## Additional Methods
Return the document frequencies calculated during fitting:
```php
public dfs() : ?array
```

### References
>- J. Martineau et al. (2009). Delta TFIDF: An Improved Feature Space for Sentiment Analysis.
>- S. Ghosh et al. (2018). Class Specific TF-IDF Boosting for Short-text Classification.
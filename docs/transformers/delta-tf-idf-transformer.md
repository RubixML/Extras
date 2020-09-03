<span style="float:right;"><a href="https://github.com/RubixML/Extras/blob/master/src/Transformers/DeltaTfIdfTransformer.php">[source]</a></span>

# Delta TF-IDF Transformer
A supervised TF-IDF (Term Frequency Inverse Document Frequency) Transformer that uses class labels to boost the TF-IDFs of terms by how informative they are. Terms that receive the highest boost are those whose concentration is primarily in one class whereas low weighted terms are more evenly distributed among the classes.

> **Note:** This transformer assumes that its input is made up of word frequency vectors such as those produced by [Word Count Vectorizer](word-count-vectorizer.md).

**Interfaces:** [Transformer](api.md#transformer), [Stateful](api.md#stateful), [Elastic](api.md#elastic)

**Data Type Compatibility:** Continuous only

## Parameters
This transformer does not have any parameters.

## Example
```php
use Rubix\ML\Transformers\DeltaTfIdfTransformer;

$transformer = new DeltaTfIdfTransformer();
```

## Additional Methods
Return the document frequencies calculated during fitting:
```php
public dfs() : ?array
```

### References
>- J. Martineau et al. (2009). Delta TFIDF: An Improved Feature Space for Sentiment Analysis.
>- S. Ghosh et al. (2018). Class Specific TF-IDF Boosting for Short-text Classification.
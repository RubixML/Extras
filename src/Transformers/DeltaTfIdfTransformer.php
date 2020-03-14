<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\DataType;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Specifications\SamplesAreCompatibleWithTransformer;
use InvalidArgumentException;
use RuntimeException;

use function is_null;

/**
 * Delta TF-IDF Transformer
 *
 * A supervised TF-IDF (Term Frequency Inverse Document Frequency) Transformer that
 * uses class labels to boost the TF-IDFs of terms by how informative they are. Terms
 * that receive the highest boost are those whose concentration is primarily in one
 * class whereas low weighted terms are more evenly distributed among the classes.
 *
 * > **Note**: This transformer assumes that its input is made up of word frequency
 * vectors such as those created by the Word Count Vectorizer.
 *
 * References:
 * [1] J. Martineau et al. (2009). Delta TFIDF: An Improved Feature Space for
 * Sentiment Analysis.
 * [2] S. Ghosh et al. (2018). Class Specific TF-IDF Boosting for Short-text
 * Classification.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class DeltaTfIdfTransformer implements Elastic
{
    /**
     * The class specific term frequencies of each word i.e. the number of
     * times a word appears in the context of a class label.
     *
     * @var array[]|null
     */
    protected $tfs;

    /**
     * The document frequencies of each word i.e. the number of times a word
     * appeared in a document given the entire corpus.
     *
     * @var int[]|null
     */
    protected $dfs;

    /**
     * The number of times a word appears throughout the entire corpus.
     *
     * @var int[]
     */
    protected $totals = [
        //
    ];

    /**
     * The number of documents that have been fitted so far.
     *
     * @var int|null
     */
    protected $n;

    /**
     * The inverse document frequency values of each feature column.
     *
     * @var float[]|null
     */
    protected $idfs;

    /**
     * The entropy for each term.
     *
     * @var float[]|null
     */
    protected $entropies;

    /**
     * Return the data types that this transformer is compatible with.
     *
     * @return \Rubix\ML\DataType[]
     */
    public function compatibility() : array
    {
        return [
            DataType::continuous(),
        ];
    }

    /**
     * Is the transformer fitted?
     *
     * @return bool
     */
    public function fitted() : bool
    {
        return $this->idfs and $this->entropies;
    }

    /**
     * Fit the transformer to the dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \InvalidArgumentException
     */
    public function fit(Dataset $dataset) : void
    {
        if (!$dataset instanceof Labeled) {
            throw new InvalidArgumentException('This transformer requires a'
                . ' labeled training set.');
        }

        $classes = $dataset->possibleOutcomes();

        $ones = array_fill(0, $dataset->numColumns(), 1);

        $this->tfs = array_fill_keys($classes, $ones);
        $this->dfs = $this->totals = $ones;
        $this->n = 1;

        $this->update($dataset);
    }

    /**
     * Update the fitting of the transformer.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     */
    public function update(Dataset $dataset) : void
    {
        if (!$dataset instanceof Labeled) {
            throw new InvalidArgumentException('This transformer requires a'
                . ' labeled training set.');
        }

        SamplesAreCompatibleWithTransformer::check($dataset, $this);

        if (is_null($this->tfs) or is_null($this->dfs)) {
            $this->fit($dataset);

            return;
        }

        foreach ($dataset->stratify() as $class => $stratum) {
            $tfs = $this->tfs[$class];

            foreach ($stratum->samples() as $sample) {
                foreach ($sample as $column => $value) {
                    if ($value > 0) {
                        $tfs[$column] += $value;

                        ++$this->dfs[$column];

                        $this->totals[$column] += $value;
                    }
                }
            }

            $this->tfs[$class] = $tfs;
        }

        $this->n += $dataset->numRows();

        $idfs = [];

        foreach ($this->dfs as $df) {
            $idfs[] = 1.0 + log($this->n / $df);
        }

        $entropies = array_fill(0, count($this->totals), 0.0);

        foreach ($this->tfs as $tfs) {
            foreach ($tfs as $column => $tf) {
                $delta = $tf / $this->totals[$column];

                $entropies[$column] += -$delta * log($delta);
            }
        }

        $this->idfs = $idfs;
        $this->entropies = $entropies;
    }

    /**
     * Transform the dataset in place.
     *
     * @param array[] $samples
     * @throws \RuntimeException
     */
    public function transform(array &$samples) : void
    {
        if (is_null($this->idfs) or is_null($this->entropies)) {
            throw new RuntimeException('Transformer has not been fitted.');
        }

        foreach ($samples as &$sample) {
            foreach ($sample as $column => &$value) {
                if ($value > 0) {
                    $value *= $this->idfs[$column];
                    $value += $this->entropies[$column];
                }
            }
        }
    }
}

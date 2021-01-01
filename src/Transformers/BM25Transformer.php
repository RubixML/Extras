<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\DataType;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Specifications\SamplesAreCompatibleWithTransformer;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;

use function is_null;

/**
 * BM25 Transformer
 *
 * BM25 is a sublinear term frequency weighting scheme that takes term frequency (TF)
 * saturation and document length into account.
 *
 * > **Note**: BM25 Transformer assumes that its inputs are made up of token frequency
 * vectors such as those created by the Word Count Vectorizer.
 *
 * References:
 * [1] S. Robertson et al. (2009). The Probabilistic Relevance Framework: BM25 and Beyond.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class BM25Transformer implements Transformer, Stateful, Elastic
{
    /**
     * The term frequency (TF) saturation factor. Lower values will cause TF to saturate quicker.
     *
     * @var float
     */
    protected $alpha;

    /**
     * The importance of document length in normalizing the term frequency.
     *
     * @var float
     */
    protected $beta;

    /**
     * The document frequencies of each word i.e. the number of times a word
     * appeared in a document given the entire corpus.
     *
     * @var int[]|null
     */
    protected $dfs;

    /**
     * The inverse document frequency values for each feature column.
     *
     * @var float[]|null
     */
    protected $idfs;

    /**
     * The number of tokens fitted so far.
     *
     * @var int|null
     */
    protected $tokenCount;

    /**
     * The number of documents (samples) that have been fitted so far.
     *
     * @var int|null
     */
    protected $n;

    /**
     * The average token count per document.
     *
     * @var float|null
     */
    protected $averageDocumentLength;

    /**
     * @param float $alpha
     * @param float $beta
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     */
    public function __construct(float $alpha = 1.2, float $beta = 0.75)
    {
        if ($alpha < 0.0) {
            throw new InvalidArgumentException('Term frequency decay'
                . " must be greater than 0, $alpha given.");
        }

        if ($beta < 0.0 or $beta > 1.0) {
            throw new InvalidArgumentException('Beta must be between'
                . " 0 and 1, $beta given.");
        }

        $this->alpha = $alpha;
        $this->beta = $beta;
    }

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
        return $this->idfs and $this->averageDocumentLength;
    }

    /**
     * Return the document frequencies calculated during fitting.
     *
     * @return int[]|null
     */
    public function dfs() : ?array
    {
        return $this->dfs;
    }

    /**
     * Return the average number of tokens per document.
     *
     * @return float|null
     */
    public function averageDocumentLength() : ?float
    {
        return $this->averageDocumentLength;
    }

    /**
     * Fit the transformer to a dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     */
    public function fit(Dataset $dataset) : void
    {
        $this->dfs = array_fill(0, $dataset->numColumns(), 1);
        $this->tokenCount = 0;
        $this->n = 1;

        $this->update($dataset);
    }

    /**
     * Update the fitting of the transformer.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     */
    public function update(Dataset $dataset) : void
    {
        SamplesAreCompatibleWithTransformer::with($dataset, $this)->check();

        if (is_null($this->dfs) or is_null($this->n)) {
            $this->fit($dataset);

            return;
        }

        foreach ($dataset->samples() as $sample) {
            foreach ($sample as $column => $tf) {
                if ($tf > 0) {
                    ++$this->dfs[$column];

                    $this->tokenCount += $tf;
                }
            }
        }

        $this->n += $dataset->numRows();

        $this->averageDocumentLength = $this->tokenCount / $this->n;

        $idfs = [];

        foreach ($this->dfs as $df) {
            $idfs[] = log(1.0 + ($this->n - $df + 0.5) / ($df + 0.5));
        }

        $this->idfs = $idfs;
    }

    /**
     * Transform the dataset in place.
     *
     * @param array[] $samples
     * @throws \Rubix\ML\Exceptions\RuntimeException
     */
    public function transform(array &$samples) : void
    {
        if (is_null($this->idfs) or is_null($this->averageDocumentLength)) {
            throw new RuntimeException('Transformer has not been fitted.');
        }

        foreach ($samples as &$sample) {
            $delta = array_sum($sample) / $this->averageDocumentLength;

            $delta = 1.0 - $this->beta + $this->beta * $delta;

            $delta *= $this->alpha;

            foreach ($sample as $column => &$tf) {
                if ($tf > 0) {
                    $tf /= $tf + $delta;
                    $tf *= $this->idfs[$column];
                }
            }
        }
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return "BM25 Transformer (alpha: {$this->alpha}, beta: {$this->beta})";
    }
}

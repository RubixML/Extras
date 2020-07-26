<?php

namespace Rubix\ML\Other\Tokenizers;

use Wamania\Snowball\StemmerFactory;

/**
 * Word Stemmer
 *
 * Word Stemmer reduces inflected and derived words to their root form.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class WordStemmer extends Word
{
    /**
     * The word stemmer.
     *
     * @var \Wamania\Snowball\Stemmer\Stemmer
     */
    protected $stemmer;

    /**
     * @param string $language
     */
    public function __construct(string $language)
    {
        $this->stemmer = StemmerFactory::create($language);
    }

    /**
     * Tokenize a block of text.
     *
     * @param string $string
     * @return string[]
     */
    public function tokenize(string $string) : array
    {
        return array_map([$this->stemmer, 'stem'], parent::tokenize($string));
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'Word Stemmer';
    }
}

<?php

namespace Rubix\ML\Persisters;

use Rubix\ML\Encoding;
use Rubix\ML\Helpers\Params;
use Rubix\ML\Exceptions\RuntimeException;
use League\Flysystem\FilesystemOperator;
use League\Flysystem\FilesystemException;

if (interface_exists('League\Flysystem\FilesystemOperator')) {
    /**
     * Flysystem
     *
     * Flysystem is a filesystem library providing a unified storage interface and abstraction layer.
     * It enables access to many different storage backends such as Local, Amazon S3, FTP, and more.
     *
     * > **Note:** The Flysystem persister is designed to work with Flysystem version 2.0.
     *
     * @see https://flysystem.thephpleague.com
     *
     * @category    Machine Learning
     * @package     Rubix/ML
     * @author      Chris Simpson
     */
    class FlysystemV2 implements Persister
    {
        /**
         * The extension to give files created as part of a persistable's save history.
         *
         * @var string
         */
        public const HISTORY_EXT = 'old';

        /**
         * The path to the model file on the filesystem.
         *
         * @var string
         */
        protected $path;

        /**
         * The filesystem implementation providing access to your backend storage.
         *
         * @var \League\Flysystem\FilesystemOperator
         */
        protected $filesystem;

        /**
         * Should we keep a history of past saves?
         *
         * @var bool
         */
        protected $history;

        /**
         * @param string $path
         * @param \League\Flysystem\FilesystemOperator $filesystem
         * @param bool $history
         */
        public function __construct(
            string $path,
            FilesystemOperator $filesystem,
            bool $history = false
        ) {
            $this->path = $path;
            $this->filesystem = $filesystem;
            $this->history = $history;
        }

        /**
         * Save an encoding.
         *
         * @param \Rubix\ML\Encoding $encoding
         *
         * @throws \Rubix\ML\Exceptions\RuntimeException
         */
        public function save(Encoding $encoding) : void
        {
            if ($this->history and $this->filesystem->fileExists($this->path)) {
                $timestamp = time();

                $filename = "{$this->path}-$timestamp." . self::HISTORY_EXT;

                $num = 0;

                while ($this->filesystem->fileExists($filename)) {
                    $filename = "{$this->path}-$timestamp-" . ++$num . '.' . self::HISTORY_EXT;
                }

                try {
                    $this->filesystem->move($this->path, $filename);
                } catch (FilesystemException $exception) {
                    throw new RuntimeException("Failed to create history file '$filename'.");
                }
            }

            try {
                $this->filesystem->write($this->path, $encoding);
            } catch (FilesystemException $exception) {
                throw new RuntimeException('Could not write to filesystem.');
            }
        }

        /**
         * Load a persisted encoding.
         *
         * @throws \Rubix\ML\Exceptions\RuntimeException
         * @return \Rubix\ML\Encoding
         */
        public function load() : Encoding
        {
            if (!$this->filesystem->fileExists($this->path)) {
                throw new RuntimeException("File does not exist at {$this->path}.");
            }

            try {
                $data = $this->filesystem->read($this->path);
            } catch (FilesystemException $exception) {
                throw new RuntimeException("Error reading data from {$this->path}.");
            }

            $encoding = new Encoding($data);

            if ($encoding->bytes() === 0) {
                throw new RuntimeException("File at {$this->path} does not contain any data.");
            }

            return $encoding;
        }

        /**
         * Return the string representation of the object.
         *
         * @return string
         */
        public function __toString() : string
        {
            return "Flysystem (path: {$this->path}, filesystem: "
                . Params::toString($this->filesystem) . ', history: '
                . Params::toString($this->history) . ')';
        }
    }
}

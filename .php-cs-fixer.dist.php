<?php

use PhpCsFixer\Finder;
use PhpCsFixer\Config;

$finder = Finder::create()->in(__DIR__)
    ->exclude('lib')
    ->exclude('ext')
    ->exclude('docs');

$config = new Config();

return $config->setRules([
    '@PSR2' => true,
    'align_multiline_comment' => true,
    'array_syntax' => ['syntax' => 'short'],
    'backtick_to_shell_exec' => true,
    'binary_operator_spaces' => true,
    'blank_line_after_namespace' => true,
    'blank_line_after_opening_tag' => true,
    'blank_line_before_statement' => [
        'statements' => [
            'break', 'case', 'continue', 'declare', 'default', 'do', 'for',
            'if', 'foreach', 'return', 'switch', 'try', 'while',
        ],
    ],
    'braces' => [
        'allow_single_line_closure' => false,
        'position_after_control_structures' => 'same',
        'position_after_functions_and_oop_constructs' => 'next',
    ],
    'cast_spaces' => ['space' => 'single'],
    'class_attributes_separation' => true,
    'combine_consecutive_issets' => true,
    'combine_consecutive_unsets' => true,
    'compact_nullable_typehint' => true,
    'concat_space' => ['spacing' => 'one'],
    'fully_qualified_strict_types' => true,
    'function_typehint_space' => true,
    'increment_style' => ['style' => 'pre'],
    'linebreak_after_opening_tag' => true,
    'list_syntax' => ['syntax' => 'short'],
    'lowercase_cast' => true,
    'lowercase_static_reference' => true,
    'magic_constant_casing' => true,
    'magic_method_casing' => true,
    'multiline_comment_opening_closing' => true,
    'multiline_whitespace_before_semicolons' => [
        'strategy' => 'no_multi_line',
    ],
    'native_function_casing' => true,
    'native_function_type_declaration_casing' => true,
    'new_with_braces' => true,
    'no_alternative_syntax' => true,
    'no_blank_lines_after_class_opening' => true,
    'no_blank_lines_after_phpdoc' => true,
    'no_empty_statement' => true,
    'no_extra_blank_lines' => true,
    'no_leading_import_slash' => true,
    'no_leading_namespace_whitespace' => true,
    'no_mixed_echo_print' => ['use' => 'echo'],
    'no_null_property_initialization' => true,
    'no_short_bool_cast' => true,
    'no_singleline_whitespace_before_semicolons' => true,
    'no_spaces_around_offset' => true,
    'no_superfluous_phpdoc_tags' => false,
    'no_superfluous_elseif' => true,
    'no_trailing_comma_in_singleline' => true,
    'no_unneeded_control_parentheses' => true,
    'no_unneeded_curly_braces' => true,
    'no_unset_cast' => true,
    'no_unused_imports' => true,
    'no_useless_else' => true,
    'no_useless_return' => true,
    'no_whitespace_before_comma_in_array' => true,
    'no_whitespace_in_blank_line' => true,
    'normalize_index_brace' => true,
    'nullable_type_declaration_for_default_null_value' => true,
    'object_operator_without_whitespace' => true,
    'ordered_class_elements' => [
        'order' => [
            'use_trait', 'constant_public', 'constant_protected',
            'constant_private', 'property_public_static', 'property_protected_static',
            'property_private_static', 'property_public', 'property_protected',
            'property_private', 'method_public_static', 'method_protected_static',
            'method_private_static', 'construct', 'destruct', 'phpunit',
            'method_public', 'method_protected', 'method_private', 'magic',
        ],
        'sort_algorithm' => 'none',
    ],
    'php_unit_fqcn_annotation' => true,
    'php_unit_method_casing' => ['case' => 'camel_case'],
    'phpdoc_add_missing_param_annotation' => ['only_untyped' => false],
    'phpdoc_align' => ['align' => 'left'],
    'phpdoc_line_span' => [
        'const' => 'multi',
        'method' => 'multi',
        'property' => 'multi',
    ],
    'phpdoc_no_access' => true,
    'phpdoc_no_empty_return' => true,
    'phpdoc_no_useless_inheritdoc' => true,
    'phpdoc_order' => true,
    'phpdoc_scalar' => true,
    'phpdoc_single_line_var_spacing' => true,
    'phpdoc_to_comment' => false,
    'phpdoc_trim' => true,
    'phpdoc_trim_consecutive_blank_line_separation' => true,
    'phpdoc_var_without_name' => true,
    'protected_to_private' => true,
    'return_assignment' => false,
    'return_type_declaration' => ['space_before' => 'one'],
    'semicolon_after_instruction' => true,
    'short_scalar_cast' => true,
    'simplified_null_return' => true,
    'single_blank_line_before_namespace' => true,
    'single_quote' => true,
    'single_line_comment_style' => true,
    'ternary_operator_spaces' => true,
    'ternary_to_null_coalescing' => true,
    'trim_array_spaces' => true,
    'unary_operator_spaces' => true,
    'whitespace_after_comma_in_array' => true,
])->setFinder($finder);

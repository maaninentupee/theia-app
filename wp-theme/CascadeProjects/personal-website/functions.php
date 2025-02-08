<?php
/**
 * Magic Jim's Personal Website Theme Functions
 */

if (!defined('ABSPATH')) {
    exit; // Exit if accessed directly
}

// Theme Setup
function magicjim_theme_setup() {
    // Add theme support
    add_theme_support('title-tag');
    add_theme_support('post-thumbnails');
    add_theme_support('custom-logo');
    add_theme_support('html5', array(
        'search-form',
        'comment-form',
        'comment-list',
        'gallery',
        'caption',
    ));

    // Register navigation menus
    register_nav_menus(array(
        'primary-menu' => __('Primary Menu', 'magicjim'),
        'footer-menu' => __('Footer Menu', 'magicjim'),
    ));
}
add_action('after_setup_theme', 'magicjim_theme_setup');

// Enqueue scripts and styles
function magicjim_scripts() {
    wp_enqueue_style('magicjim-style', get_stylesheet_uri(), array(), '1.0.0');
    wp_enqueue_script('magicjim-script', get_template_directory_uri() . '/script.js', array(), '1.0.0', true);
}
add_action('wp_enqueue_scripts', 'magicjim_scripts');

// Customizer settings
function magicjim_customize_register($wp_customize) {
    // Site Identity Section
    $wp_customize->add_setting('site_title', array(
        'default' => 'Magic Jim\'s Website',
        'sanitize_callback' => 'sanitize_text_field',
    ));
    
    $wp_customize->add_setting('site_description', array(
        'default' => 'Your go-to place for all things Magic Jim!',
        'sanitize_callback' => 'sanitize_text_field',
    ));

    // About Section
    $wp_customize->add_section('about_section', array(
        'title' => __('About Section', 'magicjim'),
        'priority' => 30,
    ));

    $wp_customize->add_setting('about_title', array(
        'default' => 'About Me',
        'sanitize_callback' => 'sanitize_text_field',
    ));

    $wp_customize->add_setting('about_content', array(
        'default' => 'I\'m Magic Jim, and I am a passionate individual with a love for magic and creativity.',
        'sanitize_callback' => 'sanitize_textarea_field',
    ));

    // Portfolio Section
    $wp_customize->add_section('portfolio_section', array(
        'title' => __('Portfolio Section', 'magicjim'),
        'priority' => 40,
    ));

    $wp_customize->add_setting('portfolio_title', array(
        'default' => 'My Work',
        'sanitize_callback' => 'sanitize_text_field',
    ));

    // Social Media Links
    $wp_customize->add_section('social_links', array(
        'title' => __('Social Media Links', 'magicjim'),
        'priority' => 50,
    ));

    $wp_customize->add_setting('social_facebook', array(
        'default' => '',
        'sanitize_callback' => 'esc_url_raw',
    ));

    $wp_customize->add_setting('social_twitter', array(
        'default' => '',
        'sanitize_callback' => 'esc_url_raw',
    ));

    $wp_customize->add_setting('social_instagram', array(
        'default' => '',
        'sanitize_callback' => 'esc_url_raw',
    ));

    // Add controls for all settings
    $wp_customize->add_control('about_title', array(
        'label' => __('About Title', 'magicjim'),
        'section' => 'about_section',
        'type' => 'text',
    ));

    $wp_customize->add_control('about_content', array(
        'label' => __('About Content', 'magicjim'),
        'section' => 'about_section',
        'type' => 'textarea',
    ));

    $wp_customize->add_control('portfolio_title', array(
        'label' => __('Portfolio Title', 'magicjim'),
        'section' => 'portfolio_section',
        'type' => 'text',
    ));

    $wp_customize->add_control('social_facebook', array(
        'label' => __('Facebook URL', 'magicjim'),
        'section' => 'social_links',
        'type' => 'url',
    ));

    $wp_customize->add_control('social_twitter', array(
        'label' => __('Twitter URL', 'magicjim'),
        'section' => 'social_links',
        'type' => 'url',
    ));

    $wp_customize->add_control('social_instagram', array(
        'label' => __('Instagram URL', 'magicjim'),
        'section' => 'social_links',
        'type' => 'url',
    ));
}
add_action('customize_register', 'magicjim_customize_register');

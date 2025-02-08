<!DOCTYPE html>
<html <?php language_attributes(); ?>>
<head>
    <meta charset="<?php bloginfo('charset'); ?>">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <?php wp_head(); ?>
</head>
<body <?php body_class(); ?>>
    <?php wp_body_open(); ?>
    
    <header class="site-header">
        <div class="container">
            <h1 class="site-title">
                <a href="<?php echo esc_url(home_url('/')); ?>">
                    <?php echo get_theme_mod('site_title', 'Magic Jim\'s Website'); ?>
                </a>
            </h1>
            <p class="site-description"><?php echo get_theme_mod('site_description', 'Your go-to place for all things Magic Jim!'); ?></p>
            
            <button class="menu-toggle" aria-controls="primary-menu" aria-expanded="false">
                <span class="menu-toggle-icon"></span>
                <span class="screen-reader-text">Menu</span>
            </button>
        </div>

        <nav class="main-navigation">
            <?php
            wp_nav_menu(array(
                'theme_location' => 'primary-menu',
                'container' => false,
                'menu_class' => 'nav-menu',
                'fallback_cb' => false
            ));
            ?>
        </nav>
    </header>

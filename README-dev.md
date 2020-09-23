# SOTU - Dev Notes

Useful links
- https://towardsdatascience.com/devops-for-data-science-with-gcp-3e6b5c3dd4f6

## Installing required packages

```bash
    $  conda install humanize
```

## Deploy Steps

- Push to github with 'prod' tag
- The prod servers will refresh automatically!  :-) 

```bash
# hack hack hack
$   git commit 

$   git push origin

#  See what tags we have
$  git tag 

# increment tag
$   git tag -a r-007
#  Enter comment

$  git push origin new_tag
# $  git push origin r-007

# move the 'prod' tag forward.
# This the tag used in deployed docker models
# we need to use '-f' to forcefully move it forward

$  git tag -af prod

# need to use force to override previous tag
$  git push origin prod --force

```
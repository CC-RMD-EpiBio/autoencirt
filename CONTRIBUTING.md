# How to contribute github


## Prerequistes

Make a github user account and clone this repository https://github.com/CC-RMD-EpiBio/autoencirt.git

```
git clone https://github.com/CC-RMD-EpiBio/autoencirt.git

git fetch --all
```

## Tutorials

Github guides in general are informative: https://guides.github.com/

1. Basic Hello World with github https://guides.github.com/activities/hello-world/ (10 - 30 minutes)
2. Understanding the github flow https://guides.github.com/introduction/flow/ (10 - 30 minutes)
3. Making your code citable https://guides.github.com/activities/citable-code/
4. Documenting projects https://guides.github.com/features/wikis/


## Contributing

### Simple Merge and Pull request 

1. `git checkout -b <new branch>`
2. Make some modifications 

3. `git add <you the files you want to stage>`

4. Add a commit message `git commit`
  - If needed to reference an issue, write `Issue #<issue number>` on the bottom
  - If needed to close an issue, write `Fixes #<issue number>` on the bottom
     + close
     + closes
     + resolves
     + resolved
     + fixed

5. Squash commits `git rebase -i <hash or HEAD~2>`  
   - Use `git log` to find the hash number
   - `HEAD~#` is the last number of commits
   - the UI is interactive and help guide you to squashing your commits

6. `git fetch --all` Grab the master branch

7. Attempt to merge the master branch `git merge upstream/master`
  - See Cherry Picking when you want your commits to exist on the end

8. `git push` Push it to your git account
   - If it doesn't exist git will complain and print out a command to run
   - The command is `git push --set-upstream origin <branch_name>`
 
9. Make a pull request on github

### Contributing with cherry picking - Please follow these directions when you have a merge conflict
Most open source projects encourage contributers to cherrypick because git place the commit on the end rather than somewhere in the middle. This project do not require cherry picking except for any changes which has a large impact on other users such as refactoring or merge conflicts with pr requests.

1. `git checkout -b <new_dev_branch>`
2. Make some modifications 

3. `git add <you the files you want to stage>`

4. Add a commit message `git commit`
  - If needed to reference an issue, write `Issue #<issue number>` on the bottom
  - If needed to close an issue, write `Fixes #<issue number>` on the bottom
     + close
     + closes
     + resolves
     + resolved
     + fixed

5. Squash commits `git rebase -i <hash or HEAD~2>`  
   - Use `git log` to find the hash number
   - `HEAD~#` is the last number of commits
   - the UI is interactive and help guide you to squashing your commits

6. `git fetch --all` Grab the master branch

7. `git checkout upstream/master` Checkout the upstream master

8. `git checkout -b <cherry_pick>` Create a branch

9. `git log <new_dev_branch>`
  - You can run `git log --oneline <new_dev_branch>` for a more condensed log
10. `git cherry-pick <hash>`
  - You can pick a range of commits with `git cherry-pick <hash_A>...<hash_B>`

11. `git push --set-upstream origin <cherry_pick>`
12.  Make a pull request on github



## Keeping up with the master branch
First, rebase your commits
### Clutter commit logs with merge 'user/branch'
I would not recommend merging during development. These commands will clutter your commits with upstreams.
```
git checkout master
git fetch --all
git merge upstream/master
git push
```

### Cherry-picking
```
git checkout master
git fetch --all
git checkout upstream/master
git checkout -b <up_branch>
git log <dev_branch> ## remember the commit hash
git cherry-pick <Hash>
git push --set-upstream origin <up_branch>
## Make a pull request
```

For maintainers

## Combining two forks
When two forks are in active development, `git merge` may combine the commits in a sequential order between the two forks. In some cases, this merge will be harmless. In other cases, the files change randomly. The best solution is to cherry-pick one fork commits into the other fork. 

1. Choose a fork you want to merge into
	+ See the commits
		*`git log --left-right --graph --cherry-pick --oneline  fork1...fork2`
	+ I choose fork one
2. See fork2 commits only
	+ `git log --left-right --graph --cherry-pick --oneline  fork1...fork2 | grep '>'` 
	+ Reverse `>` when you want to see only fork 1 commits
3. `git checkout fork1`
4. `git checkout -b cherrypick-fork2`
5. `git cherry-pick <fork1-hash>`
6. `git cherry-pick <fork1-hash>`
7. ....
8. ....
9. `git cherry-pick <fork1-hash>`

### Links
https://lwn.net/Articles/328436/


## Common git commands

* Change branches `git checkout <branch name>`
* View branch `git branch`
* View all branches `git branch -a`
* Delete branches `git branch -d <branch name>`
* View log `git log`
* View commit `git status`
* Undo Staging `git reset`



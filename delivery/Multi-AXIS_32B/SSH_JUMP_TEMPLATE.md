# Jump-host transfer template (no credentials)

Keep hostnames, users, key paths, passwords and tokens in the operator's private
SSH configuration, never in this delivery or experiment logs.

```sshconfig
Host company-jump
  HostName <jump-host>
  User <account>
  IdentityFile <private-key-path>

Host company-gpu
  HostName <internal-compute-login>
  User <account>
  ProxyJump company-jump
  IdentityFile <private-key-path>
```

Copy the verified delivery from the U disk to cluster storage using the approved
company transfer method. For example, when `rsync` is permitted:

```bash
rsync -a --info=progress2 /media/USB/Multi-AXIS_32B/ \
  company-gpu:/fast/storage/Multi-AXIS_32B/
```

Run the SHA-256 verifier again on the destination before extracting, rendering,
or training.
